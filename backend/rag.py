"""
1. Ranks edge evidence units,
2. Builds visualization chunks, and scopes the KG to retrieved evidence.
"""

from __future__ import annotations

from typing import Any

try:
    from KG_generation import (
        build_knowledge_graph_from_evidence_units,
        match_bow_terms_in_texts,
        unit_candidate_texts,
    )
except ModuleNotFoundError:
    from .KG_generation import (
        build_knowledge_graph_from_evidence_units,
        match_bow_terms_in_texts,
        unit_candidate_texts,
    )


SECTION_LABELS = {
    "intro": "Introduction",
    "related_work": "Related Work",
    "method": "Methodology",
    "experiment": "Experiments & Results",
    "conclusion": "Conclusion",
}


def score_text_vs_keywords(text: str, keywords: list[str]) -> tuple[float, list[str]]:
    text_lower = text.lower()
    matched = [kw for kw in keywords if kw in text_lower]
    ratio = len(matched) / max(len(keywords), 1)
    return ratio, matched


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def rank_evidence_units(
    evidence_units: list[dict[str, Any]],
    query_keywords: list[str],
    *,
    prompt_top_k: int = 14,
    display_top_k: int = 5,
) -> dict[str, Any]:
    """
    Query-aware evidence-unit retrieval for Edge mode.

    This is not vector retrieval; it ranks SciBERT evidence units by lexical
    query overlap, phrase overlap, importance, and evidence score.
    """
    scored_units: list[tuple[float, dict[str, Any], list[str]]] = []
    for unit in evidence_units:
        evidence = str(unit.get("evidence_sentence") or "")
        phrase = unit.get("phrase", "")
        phrase_text = " ".join(
            str(item)
            for item in (
                phrase.values() if isinstance(phrase, dict) else [phrase]
            )
        )
        words = unit.get("words") or unit.get("tokens") or []
        words_text = " ".join(str(word) for word in words) if isinstance(words, list) else str(words or "")
        combined = f"{phrase_text} {words_text} {evidence}"
        kw_score, matched_kw = score_text_vs_keywords(combined, query_keywords)

        importance = as_float(unit.get("importance"))
        evidence_score = as_float(unit.get("evidence_score"))
        role = str(unit.get("role") or "")
        role_bonus = 0.08 if role in {"core_method", "result", "contribution", "finding", "objective"} else 0.0
        relevance = 0.52 * kw_score + 0.30 * importance + 0.10 * evidence_score + role_bonus
        scored_units.append((relevance, unit, matched_kw))

    scored_units.sort(key=lambda item: item[0], reverse=True)
    top_prompt = [unit for _, unit, _ in scored_units[:prompt_top_k]]

    context_chunks: list[dict[str, Any]] = []
    for relevance, unit, matched_kw in scored_units[:display_top_k]:
        matched_bow_terms = match_bow_terms_in_texts(unit_candidate_texts(unit), limit=5)
        importance = as_float(unit.get("importance"))
        context_chunks.append(
            {
                "text": unit.get("evidence_sentence", ""),
                "phrase": unit.get("phrase", ""),
                "section": unit.get("section", "intro"),
                "section_label": SECTION_LABELS.get(unit.get("section", "intro"), "Paper"),
                "role": unit.get("role", "none"),
                "score": round(relevance, 4),
                "importance": round(importance, 4),
                "boundary_score": round(as_float(unit.get("boundary_score")), 4),
                "evidence_score": round(as_float(unit.get("evidence_score")), 4),
                "matched_keywords": matched_kw,
                "matched_bow_terms": matched_bow_terms,
                "source": "edge",
            }
        )

    kg_units = top_prompt or evidence_units[:prompt_top_k]
    knowledge_graph = build_knowledge_graph_from_evidence_units(
        kg_units,
        section_notes=None,
        term_limit=12,
        edge_limit=24,
    )

    return {
        "prompt_evidence_units": top_prompt,
        "context_chunks": context_chunks,
        "knowledge_graph": knowledge_graph,
    }