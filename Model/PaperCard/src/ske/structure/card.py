from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from ske.data.bow_vocab import BowVocabulary
from ske.data.text_utils import normalize_text
from ske.infer import build_inference_windows, coverage_rerank, extract_candidates_from_bio
from ske.modeling import ScientificKeyphraseExtractor
from ske.structure.data import StructureFeatureConfig, build_structure_features, collate_structure_features
from ske.structure.modeling import StructuredPaperModel, move_structure_batch_to_device

from .roles import infer_importance, infer_role_from_sentence
from .schema import CANONICAL_SECTIONS, ID_TO_ROLE, ConceptUnit, PaperCard, StructuredDocument
from .sectioning import section_document
from .verbalize import build_structured_summary


SECTION_NOTE_ROLES = {
    "intro": {"background", "problem", "motivation", "objective"},
    "related_work": {"prior_work", "limitation", "gap", "comparison", "problem"},
    "method": {"core_method", "component", "mechanism", "process"},
    "experiment": {"dataset", "metric", "baseline", "result", "ablation"},
    "conclusion": {"contribution", "finding", "limitation", "future_work", "result"},
}

SECTION_GOOD_CUES = {
    "intro": ("we propose", "we introduce", "in this paper", "in this work", "challenge", "limitation", "problem"),
    "related_work": ("previous", "prior", "existing", "recent", "however", "limited", "recurrent", "convolutional", "baseline"),
    "method": ("architecture", "model", "framework", "encoder", "decoder", "attention", "we use", "we employ", "we compute"),
    "experiment": ("dataset", "benchmark", "bleu", "accuracy", "f1", "outperform", "achieve", "trained", "fine-tune", "table"),
    "conclusion": ("in this work", "we presented", "we showed", "we achieve", "future", "extend", "contribution"),
}

SECTION_BAD_CUES = {
    "intro": ("table", "bleu", "accuracy", "f1", "outperform", "dataset", "training took"),
    "related_work": ("we propose", "we introduce", "our model achieves", "we achieve"),
    "method": ("leaderboard", "outperform", "test f1", "bleu score", "state-of-the-art results"),
    "experiment": ("equal contribution", "listing order", "code is available"),
    "conclusion": ("input-input", "eos pad", "figure", "table", "cls tok", "token embeddings"),
}

SECTION_BACKFILL_CUES = {
    "related_work": (
        "previous",
        "prior",
        "existing",
        "recent",
        "earlier",
        "baseline",
        "however",
        "limited",
        "unlike",
        "compared with",
        "state-of-the-art",
    ),
    "method": (
        "we propose",
        "we introduce",
        "our method",
        "our model",
        "architecture",
        "framework",
        "encoder",
        "decoder",
        "objective",
        "loss",
    ),
    "experiment": (
        "experiment",
        "evaluate",
        "dataset",
        "benchmark",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "bleu",
        "rouge",
        "table",
        "outperform",
        "achieve",
        "results show",
    ),
    "conclusion": (
        "in this work",
        "in this paper",
        "we presented",
        "we introduced",
        "we demonstrated",
        "demonstrated the viability",
        "this strong evidence shows",
        "we showed",
        "we have shown",
        "we conclude",
        "future work",
        "represents a",
        "our major contribution",
    ),
}

LOW_VALUE_PHRASES = {
    "base model",
    "large model",
    "small model",
    "beam size",
    "beam search",
    "batch size",
    "learning rate",
    "dropout",
    "dropout rate",
    "hidden layer",
    "tagging task",
    "downstream task",
    "feature-based approach",
    "single model",
    "method error",
    "openai",
    "pre",
    "squ",
}

_TOKENIZER_CACHE: dict[str, Any] = {}
_STRUCTURE_MODEL_CACHE: dict[tuple[str, str], StructuredPaperModel] = {}
_KEYPHRASE_MODEL_CACHE: dict[tuple[str, str], ScientificKeyphraseExtractor] = {}


@dataclass
class SentencePrediction:
    role: str
    role_score: float
    evidence_score: float
    importance: float


@torch.no_grad()
def build_paper_card(
    text: str,
    keyword_checkpoint: str | Path,
    structured_checkpoint: str | Path | None = None,
    title: str = "",
    bow_csv: str | Path | None = None,
    top_k_keyphrases: int = 28,
    device_name: str = "auto",
) -> PaperCard:
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu")
    if device_name != "auto":
        device = torch.device(device_name)
    sentences = section_document(text)
    document = StructuredDocument(
        doc_id=title or "paper",
        title=title,
        sentences=sentences,
        source="inference",
    )
    sentence_predictions = predict_sentence_structure(document, structured_checkpoint, device)
    candidate_pool_size = max(top_k_keyphrases * 4, top_k_keyphrases)
    keyphrase_candidates = extract_keyphrase_candidates(
        text,
        keyword_checkpoint,
        bow_csv=bow_csv,
        top_k_keyphrases=candidate_pool_size,
        device=device,
    )
    units = concept_units_from_candidates(keyphrase_candidates, document, sentence_predictions, max_units=candidate_pool_size)
    section_notes = select_section_notes(document, sentence_predictions)
    section_notes = backfill_missing_section_notes(document, sentence_predictions, section_notes)
    card = PaperCard(doc_id=document.doc_id, title=title, units=units, section_notes=section_notes)
    card.summary = build_structured_summary(card)
    return card


@torch.no_grad()
def predict_sentence_structure(
    document: StructuredDocument,
    structured_checkpoint: str | Path | None,
    device: torch.device,
) -> dict[int, SentencePrediction]:
    if not structured_checkpoint:
        return heuristic_sentence_predictions(document)
    checkpoint = Path(structured_checkpoint)
    tokenizer = load_cached_tokenizer(checkpoint)
    model = load_cached_structure_model(checkpoint, device)
    model.eval()
    record = document.to_dict()
    features = build_structure_features(record, tokenizer, StructureFeatureConfig())
    batch = collate_structure_features([features], pad_token_id=tokenizer.pad_token_id or 0)
    batch = move_structure_batch_to_device(batch, device)
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        token_type_ids=batch["token_type_ids"],
        section_token_ids=batch["section_token_ids"],
        cls_positions=batch["cls_positions"],
        sentence_mask=batch["sentence_mask"],
    )
    role_probs = torch.softmax(outputs["role_logits"][0], dim=-1).detach().cpu()
    evidence_probs = outputs["evidence_probs"][0].detach().cpu().tolist()
    importance_scores = outputs["importance_scores"][0].detach().cpu().tolist()
    predictions: dict[int, SentencePrediction] = {}
    for local_idx, sentence_idx in enumerate(features["sentence_indices"]):
        role_id = int(role_probs[local_idx].argmax().item())
        role_score = float(role_probs[local_idx, role_id].item())
        role = ID_TO_ROLE.get(role_id, "none")
        if role == "none":
            role, rule_score = infer_role_from_sentence(document.sentences[sentence_idx].text, document.sentences[sentence_idx].section)
            role_score = min(role_score, rule_score)
        predictions[int(sentence_idx)] = SentencePrediction(
            role=role,
            role_score=role_score,
            evidence_score=float(evidence_probs[local_idx]),
            importance=float(importance_scores[local_idx]),
        )
    for sentence in document.sentences:
        if sentence.sentence_index not in predictions:
            role, role_score = infer_role_from_sentence(sentence.text, sentence.section)
            importance = infer_importance(sentence.text, role)
            predictions[sentence.sentence_index] = SentencePrediction(
                role,
                role_score,
                heuristic_evidence_score(sentence.text, role, importance),
                importance,
            )
    for sentence_idx, prediction in list(predictions.items()):
        if sentence_idx < len(document.sentences):
            sentence = document.sentences[sentence_idx]
            prediction.evidence_score = calibrated_evidence_score(sentence.text, prediction.role, prediction.importance, prediction.evidence_score)
    return predictions


def heuristic_sentence_predictions(document: StructuredDocument) -> dict[int, SentencePrediction]:
    predictions: dict[int, SentencePrediction] = {}
    for sentence in document.sentences:
        role, role_score = infer_role_from_sentence(sentence.text, sentence.section)
        importance = infer_importance(sentence.text, role)
        evidence_score = heuristic_evidence_score(sentence.text, role, importance)
        predictions[sentence.sentence_index] = SentencePrediction(role, role_score, evidence_score, importance)
    return predictions


@torch.no_grad()
def extract_keyphrase_candidates(
    text: str,
    keyword_checkpoint: str | Path,
    bow_csv: str | Path | None,
    top_k_keyphrases: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    checkpoint = Path(keyword_checkpoint)
    tokenizer = load_cached_tokenizer(checkpoint)
    model = load_cached_keyphrase_model(checkpoint, device)
    model.eval()
    bow_vocab = BowVocabulary.load(str(bow_csv)) if bow_csv else None
    raw_sentences = [sentence.text for sentence in section_document(text)]
    windows = build_inference_windows(raw_sentences, tokenizer, 512)
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
    return coverage_rerank(raw_candidates, top_k_keyphrases, 0.75)


def load_cached_tokenizer(checkpoint: Path) -> Any:
    key = str(checkpoint.resolve())
    tokenizer = _TOKENIZER_CACHE.get(key)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
        _TOKENIZER_CACHE[key] = tokenizer
    return tokenizer


def load_cached_structure_model(checkpoint: Path, device: torch.device) -> StructuredPaperModel:
    key = (str(checkpoint.resolve()), device_cache_key(device))
    model = _STRUCTURE_MODEL_CACHE.get(key)
    if model is None:
        model = StructuredPaperModel.load(checkpoint, map_location=device).to(device)
        _STRUCTURE_MODEL_CACHE[key] = model
    return model


def load_cached_keyphrase_model(checkpoint: Path, device: torch.device) -> ScientificKeyphraseExtractor:
    key = (str(checkpoint.resolve()), device_cache_key(device))
    model = _KEYPHRASE_MODEL_CACHE.get(key)
    if model is None:
        model = ScientificKeyphraseExtractor.load(checkpoint, map_location=device).to(device)
        _KEYPHRASE_MODEL_CACHE[key] = model
    return model


def device_cache_key(device: torch.device) -> str:
    return f"{device.type}:{device.index if device.index is not None else ''}"


def concept_units_from_candidates(
    candidates: list[dict[str, Any]],
    document: StructuredDocument,
    predictions: dict[int, SentencePrediction],
    max_units: int | None = None,
) -> list[ConceptUnit]:
    units: list[ConceptUnit] = []
    for item in candidates:
        sentence_index = int(item.get("sentence_index", 0))
        if sentence_index >= len(document.sentences):
            continue
        sentence = document.sentences[sentence_index]
        phrase = str(item.get("canonical") or item.get("surface") or "")
        if is_noisy_phrase(phrase) or is_noisy_sentence(sentence.text):
            continue
        prediction = predictions.get(sentence_index) or SentencePrediction(*infer_role_from_sentence(sentence.text, sentence.section), 0.0, 0.5)
        stage1_score = float(item.get("score", 0.0))
        importance = 0.50 * stage1_score + 0.25 * prediction.evidence_score + 0.25 * prediction.importance
        units.append(
            ConceptUnit(
                section=sentence.section,
                phrase=phrase,
                role=prediction.role,
                evidence_sentence=sentence.text,
                importance=float(max(0.0, min(1.0, importance))),
                sentence_index=sentence_index,
                stage1_score=stage1_score,
                boundary_score=float(item.get("boundary_score", item.get("score", 0.0))),
                evidence_score=prediction.evidence_score,
                sentence_importance_score=prediction.importance,
                role_score=prediction.role_score,
            )
        )
    return select_ranked_units(units, max_units or len(units))


def select_ranked_units(units: list[ConceptUnit], max_units: int) -> list[ConceptUnit]:
    ranked = sorted(units, key=lambda unit: unit.importance, reverse=True)
    selected: list[ConceptUnit] = []
    seen: set[str] = set()

    for unit in ranked:
        key = normalize_text(unit.phrase)
        if not key or key in seen:
            continue
        seen.add(key)
        selected.append(unit)
        if len(selected) >= max_units:
            break
    return selected


def select_section_notes(document: StructuredDocument, predictions: dict[int, SentencePrediction], max_notes: int = 4) -> dict[str, list[str]]:
    notes: dict[str, list[tuple[float, str]]] = {section: [] for section in CANONICAL_SECTIONS}
    for sentence in document.sentences:
        if is_noisy_sentence(sentence.text):
            continue
        prediction = predictions.get(sentence.sentence_index)
        if prediction is None:
            continue
        score = score_section_note(sentence.section, sentence.text, prediction)
        if score <= 0.05:
            continue
        notes.setdefault(sentence.section, []).append((score, sentence.text))
    return {
        section: unique_note_texts(sorted(items, key=lambda item: item[0], reverse=True), max_notes)
        for section, items in notes.items()
    }


def backfill_missing_section_notes(
    document: StructuredDocument,
    predictions: dict[int, SentencePrediction],
    notes: dict[str, list[str]],
    max_notes: int = 4,
    min_notes: int = 1,
) -> dict[str, list[str]]:
    filled = {section: list(notes.get(section, [])) for section in CANONICAL_SECTIONS}
    seen = {normalize_text(text) for section_notes in filled.values() for text in section_notes}
    total = max(len(document.sentences), 1)
    for target_section in CANONICAL_SECTIONS:
        if len(filled.get(target_section, [])) >= min_notes:
            continue
        candidates: list[tuple[float, str]] = []
        for sentence in document.sentences:
            key = normalize_text(sentence.text)
            if not key or key in seen or is_noisy_sentence(sentence.text):
                continue
            prediction = predictions.get(sentence.sentence_index)
            if prediction is None:
                continue
            score = score_cross_section_note(target_section, sentence.text, sentence.sentence_index / total, prediction)
            if score >= 0.32:
                candidates.append((score, sentence.text))
        selected = unique_note_texts(sorted(candidates, key=lambda item: item[0], reverse=True), max_notes - len(filled[target_section]))
        filled[target_section].extend(selected)
        seen.update(normalize_text(text) for text in selected)
    return filled


def score_cross_section_note(target_section: str, sentence: str, position_ratio: float, prediction: SentencePrediction) -> float:
    normalized = normalize_text(sentence)
    allowed_roles = SECTION_NOTE_ROLES.get(target_section, set())
    has_backfill_cue = any(cue in normalized for cue in SECTION_BACKFILL_CUES.get(target_section, ()))
    if target_section == "conclusion" and not has_strong_conclusion_cue(normalized):
        return 0.0
    score = 0.22 * prediction.importance + 0.22 * prediction.evidence_score
    if prediction.role in allowed_roles:
        score += 0.26
    if has_backfill_cue:
        score += 0.20
    if target_section == "conclusion" and position_ratio >= 0.68:
        score += 0.12
    if target_section == "related_work" and position_ratio <= 0.45:
        score += 0.08
    if target_section == "experiment" and position_ratio >= 0.45:
        score += 0.08
    if target_section == "method" and 0.15 <= position_ratio <= 0.70:
        score += 0.08
    if target_section != "intro" and normalized.startswith("title "):
        score -= 0.50
    if len(normalized.split()) < 10:
        score -= 0.08
    return max(0.0, min(1.0, score))


def has_strong_conclusion_cue(normalized: str) -> bool:
    starts = (
        "in this work",
        "in this paper",
        "we presented",
        "we introduced",
        "we demonstrated",
        "we showed",
        "we have shown",
        "we conclude",
        "future work",
    )
    if normalized.startswith(starts):
        return True
    strong_anywhere = (
        "our major contribution",
        "this strong evidence shows",
        "demonstrated the viability",
        "represents a significant step",
    )
    return any(cue in normalized for cue in strong_anywhere)


def score_section_note(section: str, sentence: str, prediction: SentencePrediction) -> float:
    normalized = normalize_text(sentence)
    role_bonus = 0.18 if prediction.role in SECTION_NOTE_ROLES.get(section, set()) else -0.16
    score = 0.46 * prediction.importance + 0.34 * prediction.evidence_score + role_bonus
    if any(cue in normalized for cue in SECTION_GOOD_CUES.get(section, ())):
        score += 0.12
    if any(cue in normalized for cue in SECTION_BAD_CUES.get(section, ())):
        score -= 0.22
    if len(normalized.split()) < 10:
        score -= 0.10
    return max(0.0, min(1.0, score))


def unique_note_texts(items: list[tuple[float, str]], max_notes: int) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for _, text in items:
        key = normalize_text(text)
        if not key or key in seen:
            continue
        if any(key in old or old in key for old in seen):
            continue
        seen.add(key)
        selected.append(text)
        if len(selected) >= max_notes:
            break
    return selected


def heuristic_evidence_score(sentence: str, role: str, importance: float) -> float:
    cue_score = 0.0
    lowered = sentence.lower()
    if any(cue in lowered for cue in ("we propose", "we introduce", "we present", "in this work", "our model")):
        cue_score = max(cue_score, 0.62)
    if any(cue in lowered for cue in ("results show", "outperforms", "achieves", "bleu", "accuracy", "f1")):
        cue_score = max(cue_score, 0.58)
    if any(cue in lowered for cue in ("because", "therefore", "this suggests", "we find", "we show")):
        cue_score = max(cue_score, 0.48)
    if role in {"core_method", "result", "finding", "objective"}:
        cue_score = max(cue_score, 0.35)
    if importance >= 0.72:
        cue_score = max(cue_score, 0.32)
    return min(1.0, cue_score)


def calibrated_evidence_score(sentence: str, role: str, importance: float, model_score: float) -> float:
    if is_noisy_sentence(sentence):
        return 0.0
    return max(float(model_score), heuristic_evidence_score(sentence, role, importance) * 0.75)


def is_noisy_phrase(phrase: str) -> bool:
    normalized = normalize_text(phrase)
    if not normalized or len(normalized) < 3:
        return True
    if normalized.endswith("-") or normalized.startswith("-"):
        return True
    if normalized in {"eos", "pad", "eos pad", "input input", "unk", "big", "cloud tpu"}:
        return True
    if normalized in LOW_VALUE_PHRASES:
        return True
    if normalized in {
        "bad guy",
        "redacted website",
        "sistine chapel",
        "wheelchair",
        "gang rape",
        "taskrabbit",
        "marathon",
        "power shutoff",
    }:
        return True
    if normalized.endswith(" gp"):
        return True
    tokens = normalized.split()
    if tokens and sum(token in {"eos", "pad"} for token in tokens) / len(tokens) >= 0.5:
        return True
    return False


def is_noisy_sentence(sentence: str) -> bool:
    normalized = normalize_text(sentence)
    if not normalized:
        return True
    if normalized.startswith("title ") and len(normalized.split()) <= 12:
        return True
    if normalized.startswith("title core contributors"):
        return True
    if re.search(r"\bsuch as and\b", normalized):
        return True
    noise_cues = [
        "eos pad",
        "pad pad",
        "input input layer",
        "best viewed in color",
        "attention visualizations",
        "references",
        "appendix",
        "system card",
        "equal contribution listing order",
        "work performed while",
        "code and pre-trained models are available",
        "code we used to train and evaluate",
        "cls tok",
        "e cls",
        "token embeddings",
        "choose from the following options",
        "single character a or b",
        "assistant generation",
        "attractiveness score",
        "redacted website",
        "taskrabbit worker",
        "unlicensed firearms",
        "how can i kill myself",
        "gang rape",
        "molecule search purchase patent search",
        "amazing office space",
        "administrative hr legal",
        "everyone at openai has contributed",
        "participation in this red teaming process",
        "gollo jattani",
        "sam manning",
        "red teaming process was",
        "full marathon",
        "power shutoff",
        "customers in california",
        "not officially sanctioned world record",
        "table shows example summaries",
        "example summaries generated",
        "qualitative analysis",
        "order for the chains to be able to mix",
    ]
    for cue in noise_cues:
        if cue not in normalized:
            continue
        if cue.startswith("code") and any(good in normalized for good in ("in this work", "we presented", "we show", "we achieve")):
            continue
        return True
    tokens = normalized.split()
    strong_short_cues = (
        "we propose",
        "we introduce",
        "we present",
        "we show",
        "we achieve",
        "in this work",
        "in this paper",
    )
    if len(tokens) < 8 and not any(cue in normalized for cue in strong_short_cues):
        return True
    if tokens and sum(token in {"eos", "pad"} for token in tokens) / len(tokens) > 0.08:
        return True
    return False
