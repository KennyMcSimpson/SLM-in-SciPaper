"""
CSV-grounded knowledge graph generation for SciPaper.

This module does three things:
1. Loads the global BoW CSV vocabulary and matches paper text to canonical terms.
2. Extracts lightweight semantic edges from paper evidence sentences.
3. Formats matched terminology for the RAG prompt.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BOW_VOCAB_PATH = PROJECT_DIR / "BagofWord" / "final_bow_vocabulary.csv"

_BOW_VOCAB_CACHE: list[dict[str, Any]] | None = None


RELATION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("uses", re.compile(r"\b(?:use|uses|used|using|utilize|utilizes|employ|employs|adopt|adopts|leverage|leverages)\b", re.I)),
    ("relies_on", re.compile(r"\b(?:rely|relies|relied|relying)\s+on\b", re.I)),
    ("based_on", re.compile(r"\b(?:based\s+on|built\s+on|builds\s+on|derived\s+from)\b", re.I)),
    ("contains", re.compile(r"\b(?:contain|contains|include|includes|comprise|comprises|consist|consists)\b", re.I)),
    ("improves", re.compile(r"\b(?:improve|improves|improved|outperform|outperforms|achieve|achieves|boost|boosts)\b", re.I)),
    ("reduces", re.compile(r"\b(?:reduce|reduces|reduced|lower|lowers|mitigate|mitigates)\b", re.I)),
    ("trains_on", re.compile(r"\b(?:train|trains|trained|pre-train|pre-trains|pre-trained|fine-tune|fine-tunes|fine-tuned)\s+(?:on|with|using)\b", re.I)),
    ("evaluates_with", re.compile(r"\b(?:evaluate|evaluates|evaluated|measure|measures|measured|benchmark|benchmarks)\b", re.I)),
    ("retrieves", re.compile(r"\b(?:retrieve|retrieves|retrieved|retrieval|search|searches)\b", re.I)),
    ("generates", re.compile(r"\b(?:generate|generates|generated|generation|decode|decodes|decoded)\b", re.I)),
    ("encodes", re.compile(r"\b(?:encode|encodes|encoded|represent|represents|representation|embed|embeds|embedding)\b", re.I)),
]


def normalize_term_text(text: object) -> str:
    if text is None:
        return ""
    normalized = str(text).lower().replace("\ufeff", "")
    normalized = re.sub(r"[^a-z0-9+\-]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def split_aliases(value: object) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def alias_variants(alias: str) -> set[str]:
    normalized = normalize_term_text(alias)
    variants = {normalized}
    if "-" in normalized:
        variants.add(normalized.replace("-", " "))
    if " " in normalized:
        variants.add(normalized.replace(" ", "-"))
    return {item for item in variants if item}


def contains_normalized_alias(text: str, alias: str) -> bool:
    if not text or not alias:
        return False
    return bool(re.search(rf"(?<![a-z0-9+\-]){re.escape(alias)}(?![a-z0-9+\-])", text))


def infer_bow_domain(term: dict[str, Any]) -> str:
    text = normalize_term_text(
        " ".join(
            [
                str(term.get("canonical_term", "")),
                str(term.get("display_term", "")),
                str(term.get("wiki_category", "")),
            ]
        )
    )
    rules = [
        ("NLP & LLM", ["natural language", "language model", "large language model", "bert", "transformer", "text", "token", "translation", "question answering", "embedding"]),
        ("Deep Learning", ["deep learning", "neural network", "artificial neural", "attention", "backpropagation", "convolutional", "activation", "layer"]),
        ("Information Retrieval", ["information retrieval", "retrieval", "rag", "search", "ranking", "passage", "query", "bm25"]),
        ("Optimization", ["optimization", "gradient", "loss", "optimizer", "regularization", "descent", "learning rate"]),
        ("Evaluation", ["evaluation", "metric", "benchmark", "accuracy", "bleu", "rouge", "f1"]),
        ("Computer Vision", ["computer vision", "image", "vision", "object detection", "convolutional neural network"]),
        ("Graph Learning", ["graph", "node", "edge", "knowledge graph", "graph neural"]),
        ("Reinforcement Learning", ["reinforcement", "policy", "reward", "q-learning", "actor"]),
        ("Data & Training", ["training data", "training set", "dataset", "corpus", "pre-training"]),
    ]
    for domain, cues in rules:
        if any(cue in text for cue in cues):
            return domain
    return str(term.get("wiki_category") or "Other")


def load_global_bow_vocab(csv_path: Path = DEFAULT_BOW_VOCAB_PATH) -> list[dict[str, Any]]:
    global _BOW_VOCAB_CACHE
    if _BOW_VOCAB_CACHE is not None:
        return _BOW_VOCAB_CACHE

    vocab: list[dict[str, Any]] = []
    if not csv_path.exists():
        _BOW_VOCAB_CACHE = vocab
        return vocab

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            canonical = (
                row.get("canonical_term")
                or row.get("display_term")
                or row.get("term")
                or row.get("normalized_term")
                or ""
            ).strip()
            if not canonical:
                continue

            aliases = [
                row.get("display_term", ""),
                row.get("term", ""),
                row.get("normalized_term", ""),
                row.get("canonical_term", ""),
            ]
            aliases.extend(split_aliases(row.get("matched_surface_terms")))
            aliases.extend(split_aliases(row.get("search_terms")))
            normalized_aliases = sorted(
                {variant for alias in aliases for variant in alias_variants(alias)},
                key=len,
                reverse=True,
            )
            if not normalized_aliases:
                continue

            try:
                confidence = float(row.get("confidence_score") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0

            vocab.append(
                {
                    "display_term": (row.get("display_term") or canonical).strip(),
                    "canonical_term": canonical,
                    "aliases": normalized_aliases,
                    "wiki_category": (row.get("wiki_category") or "").strip(),
                    "wiki_url": (row.get("wiki_url") or "").strip(),
                    "wikidata_id": (row.get("wikidata_id") or "").strip(),
                    "document_frequency": row.get("document_frequency") or "",
                    "total_frequency": row.get("total_frequency") or "",
                    "confidence_score": confidence,
                }
            )

    _BOW_VOCAB_CACHE = vocab
    return vocab


def unit_candidate_texts(unit: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    phrase = unit.get("phrase", "")
    if isinstance(phrase, dict):
        candidates.extend(str(phrase.get(key) or "") for key in ("surface", "canonical", "normalized"))
    else:
        candidates.append(str(phrase or ""))

    words = unit.get("words") or unit.get("tokens") or unit.get("word_tokens")
    if isinstance(words, list):
        candidates.append(" ".join(str(word) for word in words))
        candidates.extend(str(word) for word in words)
    elif isinstance(words, str):
        candidates.append(words)

    candidates.append(str(unit.get("evidence_sentence") or ""))
    return [text for text in candidates if text.strip()]


def match_bow_terms_in_texts(texts: list[str], limit: int = 12) -> list[dict[str, Any]]:
    vocab = load_global_bow_vocab()
    normalized_texts = [normalize_term_text(text) for text in texts if text]
    if not vocab or not normalized_texts:
        return []

    matches: dict[str, dict[str, Any]] = {}
    for term in vocab:
        best: tuple[float, str] | None = None
        for alias in term["aliases"]:
            for idx, text in enumerate(normalized_texts):
                if not contains_normalized_alias(text, alias):
                    continue
                phrase_bonus = 0.25 if idx < max(1, len(normalized_texts) // 3) else 0.0
                specificity = min(len(alias.split()) / 4, 0.25)
                score = min(1.0, float(term["confidence_score"]) + phrase_bonus + specificity)
                if best is None or score > best[0]:
                    best = (score, alias)
        if best is None:
            continue

        key = normalize_term_text(term["canonical_term"])
        current = matches.get(key)
        if current is None or best[0] > current["match_score"]:
            matches[key] = {
                "term": term["display_term"],
                "canonical_term": term["canonical_term"],
                "matched_alias": best[1],
                "category": term["wiki_category"],
                "domain": infer_bow_domain(term),
                "wiki_url": term["wiki_url"],
                "wikidata_id": term["wikidata_id"],
                "document_frequency": term["document_frequency"],
                "total_frequency": term["total_frequency"],
                "confidence": round(float(term["confidence_score"]), 4),
                "match_score": round(best[0], 4),
            }

    return sorted(matches.values(), key=lambda item: (item["match_score"], item["confidence"]), reverse=True)[:limit]


def extract_bow_terms_from_evidence_units(
    evidence_units: list[dict[str, Any]],
    section_notes: dict[str, Any] | None = None,
    limit: int = 12,
) -> list[dict[str, Any]]:
    texts: list[str] = []
    for unit in evidence_units:
        texts.extend(unit_candidate_texts(unit))

    if section_notes:
        for notes in section_notes.values():
            if isinstance(notes, list):
                texts.extend(str(note) for note in notes)

    return match_bow_terms_in_texts(texts, limit=limit)


def terms_in_sentence(sentence: str, vocabulary_terms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = normalize_term_text(sentence)
    hits: dict[str, dict[str, Any]] = {}
    for term in vocabulary_terms:
        aliases = term.get("aliases")
        if not aliases:
            aliases = [
                normalize_term_text(term.get("matched_alias", "")),
                normalize_term_text(term.get("canonical_term", "")),
                normalize_term_text(term.get("term", "")),
            ]
        for alias in aliases:
            if contains_normalized_alias(normalized, str(alias)):
                key = normalize_term_text(term["canonical_term"])
                hits[key] = term
                break
    return sorted(
        hits.values(),
        key=lambda item: normalized.find(
            normalize_term_text(
                item.get("matched_alias") or item.get("canonical_term") or item.get("term")
            )
        ),
    )


def infer_relations(sentence: str, source: dict[str, Any], target: dict[str, Any]) -> list[str]:
    if normalize_term_text(source["canonical_term"]) == normalize_term_text(target["canonical_term"]):
        return []

    relations: list[str] = []
    for relation, pattern in RELATION_PATTERNS:
        if pattern.search(sentence):
            relations.append(relation)
    return relations or ["co_occurs_in_evidence"]


def build_semantic_edges_from_evidence(
    evidence_units: list[dict[str, Any]],
    matched_terms: list[dict[str, Any]],
    max_edges: int = 24,
) -> list[dict[str, Any]]:
    if len(matched_terms) < 2:
        return []

    edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    for unit in evidence_units:
        sentence = str(unit.get("evidence_sentence") or "")
        if not sentence:
            continue
        hits = terms_in_sentence(sentence, matched_terms)
        if len(hits) < 2:
            continue
        for idx, source in enumerate(hits):
            for target in hits[idx + 1 :]:
                source_key = normalize_term_text(source["canonical_term"])
                target_key = normalize_term_text(target["canonical_term"])
                for relation in infer_relations(sentence, source, target):
                    edge_key = (source_key, target_key, relation)
                    if edge_key not in edges:
                        edges[edge_key] = {
                            "source": source["canonical_term"],
                            "target": target["canonical_term"],
                            "relation": relation,
                            "evidence": sentence[:260],
                            "section": unit.get("section", ""),
                            "role": unit.get("role", ""),
                            "source_type": "paper_evidence",
                        }
    return list(edges.values())[:max_edges]


def build_semantic_edges_from_texts(
    texts: list[str],
    matched_terms: list[dict[str, Any]],
    max_edges: int = 24,
) -> list[dict[str, Any]]:
    pseudo_units = [{"evidence_sentence": text, "section": "", "role": ""} for text in texts]
    return build_semantic_edges_from_evidence(pseudo_units, matched_terms, max_edges=max_edges)


def build_knowledge_graph_from_evidence_units(
    evidence_units: list[dict[str, Any]],
    section_notes: dict[str, Any] | None = None,
    term_limit: int = 12,
    edge_limit: int = 24,
) -> dict[str, Any]:
    terms = extract_bow_terms_from_evidence_units(evidence_units, section_notes=section_notes, limit=term_limit)
    edges = build_semantic_edges_from_evidence(evidence_units, terms, max_edges=edge_limit)
    return {"terms": terms, "edges": edges}


def build_knowledge_graph_from_texts(
    texts: list[str],
    term_limit: int = 12,
    edge_limit: int = 24,
) -> dict[str, Any]:
    terms = match_bow_terms_in_texts(texts, limit=term_limit)
    edges = build_semantic_edges_from_texts(texts, terms, max_edges=edge_limit)
    return {"terms": terms, "edges": edges}


def format_bow_terms_for_prompt(terms: list[dict[str, Any]]) -> str:
    if not terms:
        return ""

    lines = ["\n\n Matched from global BoW vocabulary"]
    lines.append(
        "Use these paper-specific term matches to explain important technical words when they are relevant. "
        "Treat them as terminology hints grounded by the extracted paper phrases, not as substitutes for the paper evidence."
    )
    for term in terms[:12]:
        category = f" | category: {term['category']}" if term.get("category") else ""
        domain = f" | domain: {term['domain']}" if term.get("domain") else ""
        wiki = f" | wiki: {term['wiki_url']}" if term.get("wiki_url") else ""
        lines.append(
            f"- {term['canonical_term']} (matched: {term['matched_alias']}; "
            f"confidence: {term['confidence']}; match: {term['match_score']}{domain}{category}{wiki})"
        )
    return "\n".join(lines)
