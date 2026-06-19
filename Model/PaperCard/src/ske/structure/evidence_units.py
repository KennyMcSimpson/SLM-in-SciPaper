from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ske.data.text_utils import find_normalized_phrase_spans, normalize_text, token_jaccard

from .schema import CANONICAL_SECTIONS, ConceptUnit, PaperCard, SentenceRecord
from .sectioning import section_document


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "this",
    "to",
    "we",
    "with",
}

SECTION_ALIASES = {
    "abstract": "intro",
    "introduction": "intro",
    "intro": "intro",
    "background": "intro",
    "related work": "related_work",
    "related_work": "related_work",
    "prior work": "related_work",
    "methods": "method",
    "method": "method",
    "methodology": "method",
    "experiments": "experiment",
    "experiment": "experiment",
    "evaluation": "experiment",
    "results": "experiment",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
}

ROLE_SUPPORT_TYPES = {
    "background": "background_context",
    "problem": "research_problem",
    "motivation": "research_motivation",
    "objective": "paper_objective",
    "prior_work": "related_work_claim",
    "limitation": "limitation_or_gap",
    "gap": "limitation_or_gap",
    "comparison": "method_comparison",
    "core_method": "method_mechanism",
    "component": "method_component",
    "mechanism": "method_mechanism",
    "process": "method_process",
    "dataset": "experimental_setting",
    "metric": "evaluation_metric",
    "baseline": "comparison_baseline",
    "result": "experimental_result",
    "ablation": "ablation_evidence",
    "contribution": "contribution_claim",
    "finding": "paper_finding",
    "future_work": "future_work",
}


@dataclass(frozen=True)
class SectionBowTerm:
    section: str
    original_section: str
    display_term: str
    canonical_term: str
    aliases: tuple[str, ...]
    source: str
    wiki_category: str
    wiki_url: str
    wikidata_id: str
    match_type: str
    term_type: str
    role_prior: str
    ngram_label: str
    pos_pattern: str
    is_generic: bool
    source_corpus: str
    notes: str
    document_frequency: int
    total_frequency: int
    confidence_score: float


@dataclass(frozen=True)
class SectionBowMatch:
    term: SectionBowTerm
    alias: str
    match_quality: float
    section_prior: float

    @property
    def confidence(self) -> float:
        return max(0.0, min(1.0, self.term.confidence_score * self.match_quality))


class SectionBowVocabulary:
    def __init__(self, terms: list[SectionBowTerm]) -> None:
        self.terms = terms
        self.alias_to_terms: dict[str, list[SectionBowTerm]] = defaultdict(list)
        self.canonical_section_tf: dict[str, Counter[str]] = defaultdict(Counter)
        self.section_totals: Counter[str] = Counter()
        for term in terms:
            canonical_key = normalize_text(term.canonical_term or term.display_term)
            self.canonical_section_tf[canonical_key][term.section] += max(term.total_frequency, 0)
            self.section_totals[term.section] += max(term.total_frequency, 0)
            for alias in term.aliases:
                normalized_alias = normalize_text(alias)
                if normalized_alias:
                    self.alias_to_terms[normalized_alias].append(term)

    @classmethod
    def load(cls, path: str | Path | None) -> "SectionBowVocabulary | None":
        if not path:
            return None
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        terms: list[SectionBowTerm] = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                original_section = (row.get("section_mapped") or row.get("section") or row.get("section_original") or "").strip()
                canonical = (row.get("canonical_term") or row.get("display_term") or row.get("term") or "").strip()
                aliases = collect_aliases(row, canonical)
                if not canonical or not aliases:
                    continue
                is_generic = safe_bool(row.get("is_generic"))
                term_type = (row.get("term_type") or row.get("wiki_category") or "").strip()
                terms.append(
                    SectionBowTerm(
                        section=normalize_section(original_section),
                        original_section=original_section,
                        display_term=(row.get("display_term") or canonical).strip(),
                        canonical_term=canonical,
                        aliases=tuple(aliases),
                        source=(row.get("source") or row.get("source_corpus") or "").strip(),
                        wiki_category=(row.get("wiki_category") or term_type).strip(),
                        wiki_url=(row.get("wiki_url") or "").strip(),
                        wikidata_id=(row.get("wikidata_id") or "").strip(),
                        match_type=(row.get("match_type") or "").strip(),
                        term_type=term_type,
                        role_prior=(row.get("role_prior") or "").strip(),
                        ngram_label=(row.get("ngram_len") or "").strip(),
                        pos_pattern=(row.get("pos_pattern") or "").strip(),
                        is_generic=is_generic,
                        source_corpus=(row.get("source_corpus") or "").strip(),
                        notes=(row.get("notes") or "").strip(),
                        document_frequency=safe_int(row.get("document_frequency"), 0),
                        total_frequency=safe_int(row.get("total_frequency"), 0),
                        confidence_score=infer_bow_confidence(row, is_generic),
                    )
                )
        return cls(terms)

    def match_phrase(self, phrase: str, section: str | None = None) -> SectionBowMatch | None:
        normalized_phrase = normalize_text(phrase)
        if not normalized_phrase:
            return None
        target_section = normalize_section(section)
        candidates: list[tuple[float, str, SectionBowTerm]] = []
        for alias, terms in self.alias_to_terms.items():
            quality = alias_match_quality(normalized_phrase, alias)
            if quality <= 0.0:
                continue
            for term in terms:
                score = quality
                score += 0.15 * term.confidence_score
                score += 0.10 if term.section == target_section else 0.0
                score += 0.05 * min(1.0, math.log1p(term.total_frequency) / 6.0)
                if term.is_generic:
                    score -= 0.18
                candidates.append((score, alias, term))
        if not candidates:
            return None
        _, alias, term = max(candidates, key=lambda item: item[0])
        canonical_key = normalize_text(term.canonical_term or term.display_term)
        return SectionBowMatch(
            term=term,
            alias=alias,
            match_quality=alias_match_quality(normalized_phrase, alias),
            section_prior=self.section_prior(canonical_key, target_section),
        )

    def sentence_hits(self, sentence: str, section: str | None = None, min_confidence: float = 0.65, max_hits: int = 8) -> list[SectionBowMatch]:
        target_section = normalize_section(section)
        matches: list[SectionBowMatch] = []
        seen: set[tuple[str, str]] = set()
        aliases = sorted(self.alias_to_terms, key=lambda item: (len(item.split()), len(item)), reverse=True)
        for alias in aliases:
            if not find_normalized_phrase_spans(sentence, alias):
                continue
            for term in self.alias_to_terms[alias]:
                if term.confidence_score < min_confidence:
                    continue
                key = (normalize_text(term.canonical_term), term.section)
                if key in seen:
                    continue
                seen.add(key)
                matches.append(
                    SectionBowMatch(
                        term=term,
                        alias=alias,
                        match_quality=1.0,
                        section_prior=self.section_prior(normalize_text(term.canonical_term), target_section),
                    )
                )
        matches.sort(
            key=lambda item: (
                item.term.section == target_section,
                item.confidence,
                item.term.total_frequency,
            ),
            reverse=True,
        )
        return matches[:max_hits]

    def section_prior(self, canonical_key: str, section: str | None, alpha: float = 1.0) -> float:
        target_section = normalize_section(section)
        counts = self.canonical_section_tf.get(canonical_key)
        section_count = float(counts.get(target_section, 0) if counts else 0)
        total = float(sum(counts.values()) if counts else 0)
        return (section_count + alpha) / (total + alpha * len(CANONICAL_SECTIONS))


@dataclass(frozen=True)
class EvidenceCue:
    cue_phrase: str
    evidence_type: str
    role_prior: str
    section_hint: str
    notes: str


class EvidenceCueLexicon:
    def __init__(self, cues: list[EvidenceCue]) -> None:
        self.cues = sorted(cues, key=lambda item: len(item.cue_phrase), reverse=True)

    @classmethod
    def load(cls, path: str | Path | None) -> "EvidenceCueLexicon | None":
        if not path:
            return None
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        cues: list[EvidenceCue] = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                cue = (row.get("cue_phrase") or "").strip()
                if not cue:
                    continue
                cues.append(
                    EvidenceCue(
                        cue_phrase=cue,
                        evidence_type=(row.get("evidence_type") or "").strip(),
                        role_prior=(row.get("role_prior") or "").strip(),
                        section_hint=normalize_section(row.get("section_hint")),
                        notes=(row.get("notes") or "").strip(),
                    )
                )
        return cls(cues)

    def sentence_hits(self, sentence: str, section: str | None = None, max_hits: int = 8) -> list[EvidenceCue]:
        target_section = normalize_section(section)
        hits: list[EvidenceCue] = []
        seen: set[str] = set()
        for cue in self.cues:
            normalized = normalize_text(cue.cue_phrase)
            if not normalized or normalized in seen:
                continue
            if not find_normalized_phrase_spans(sentence, normalized):
                continue
            seen.add(normalized)
            hits.append(cue)
        hits.sort(key=lambda item: (item.section_hint == target_section, len(item.cue_phrase)), reverse=True)
        return hits[:max_hits]


@dataclass(frozen=True)
class MatrixFeatureScore:
    feature_name: str
    section: str
    term: str
    frequency: float
    tfidf: float


class DocumentTermMatrix:
    def __init__(self, frequency: dict[str, float], tfidf: dict[str, float]) -> None:
        self.frequency = frequency
        self.tfidf = tfidf
        self.max_frequency = max(frequency.values(), default=0.0)
        self.max_tfidf = max(tfidf.values(), default=0.0)

    @classmethod
    def load(
        cls,
        frequency_path: str | Path | None,
        tfidf_path: str | Path | None,
        doc_id: str,
    ) -> "DocumentTermMatrix | None":
        frequency = load_matrix_row(frequency_path, doc_id)
        tfidf = load_matrix_row(tfidf_path, doc_id)
        if not frequency and not tfidf:
            return None
        return cls(frequency, tfidf)

    def feature_score(self, section: str | None, phrase: str) -> MatrixFeatureScore | None:
        target_section = normalize_section(section)
        normalized_phrase = normalize_text(phrase)
        candidates: list[tuple[float, MatrixFeatureScore]] = []
        for feature in set(self.frequency) | set(self.tfidf):
            feature_section, feature_term = split_matrix_feature(feature)
            if feature_section != target_section:
                continue
            quality = alias_match_quality(normalized_phrase, normalize_text(feature_term))
            if quality <= 0.0:
                continue
            frequency = self.frequency.get(feature, 0.0)
            tfidf = self.tfidf.get(feature, 0.0)
            candidates.append(
                (
                    quality + 0.15 * normalized_score(tfidf, self.max_tfidf) + 0.05 * normalized_score(frequency, self.max_frequency),
                    MatrixFeatureScore(feature, feature_section, feature_term, frequency, tfidf),
                )
            )
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    def top_features(self, section: str | None = None, limit: int = 20) -> list[MatrixFeatureScore]:
        target_section = normalize_section(section) if section else None
        items: list[MatrixFeatureScore] = []
        for feature in set(self.frequency) | set(self.tfidf):
            feature_section, feature_term = split_matrix_feature(feature)
            if target_section and feature_section != target_section:
                continue
            frequency = self.frequency.get(feature, 0.0)
            tfidf = self.tfidf.get(feature, 0.0)
            if frequency <= 0.0 and tfidf <= 0.0:
                continue
            items.append(MatrixFeatureScore(feature, feature_section, feature_term, frequency, tfidf))
        items.sort(key=lambda item: (item.tfidf, item.frequency), reverse=True)
        return items[:limit]


@dataclass(frozen=True)
class ExternalEvidenceCandidate:
    document_id: str
    section_original: str
    section: str
    sentence: str
    evidence_type: str
    matched_cue: str
    role_prior: str
    confidence: float


class ExternalEvidenceIndex:
    def __init__(self, candidates: list[ExternalEvidenceCandidate]) -> None:
        self.candidates = candidates

    @classmethod
    def load(cls, path: str | Path | None, doc_id: str) -> "ExternalEvidenceIndex | None":
        if not path:
            return None
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        candidates: list[ExternalEvidenceCandidate] = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if (row.get("document_id") or "").strip() != doc_id:
                    continue
                sentence = (row.get("sentence") or "").strip()
                if not sentence:
                    continue
                candidates.append(
                    ExternalEvidenceCandidate(
                        document_id=(row.get("document_id") or "").strip(),
                        section_original=(row.get("section_original") or "").strip(),
                        section=normalize_section(row.get("section_mapped") or row.get("section_original")),
                        sentence=sentence,
                        evidence_type=(row.get("evidence_type") or "").strip(),
                        matched_cue=(row.get("matched_cue") or "").strip(),
                        role_prior=(row.get("role_prior") or "").strip(),
                        confidence=safe_float(row.get("confidence"), 0.0),
                    )
                )
        return cls(candidates)

    def top_candidates(self, limit: int = 25) -> list[dict[str, Any]]:
        candidates = sorted(self.candidates, key=lambda item: item.confidence, reverse=True)
        return [
            {
                "section": item.section,
                "section_original": item.section_original,
                "sentence": item.sentence,
                "evidence_type": item.evidence_type,
                "matched_cue": item.matched_cue,
                "role_prior": item.role_prior,
                "confidence": round_float(item.confidence),
            }
            for item in candidates[:limit]
        ]

    def sentence_overlap(self, sentence: str) -> dict[str, Any]:
        normalized = normalize_text(sentence)
        if not normalized:
            return {"matched": False}
        best: tuple[float, ExternalEvidenceCandidate] | None = None
        for item in self.candidates:
            score = token_jaccard(normalized, item.sentence)
            if best is None or score > best[0]:
                best = (score, item)
        if best is None or best[0] < 0.68:
            return {"matched": False}
        item = best[1]
        return {
            "matched": True,
            "overlap": round_float(best[0]),
            "evidence_type": item.evidence_type,
            "matched_cue": item.matched_cue,
            "role_prior": item.role_prior,
            "confidence": round_float(item.confidence),
        }


def build_evidence_units_payload(
    card: PaperCard,
    text: str,
    source_path: str | Path | None = None,
    keyword_checkpoint: str | Path | None = None,
    structured_checkpoint: str | Path | None = None,
    bow_csv: str | Path | None = None,
    section_bow_csv: str | Path | None = None,
    term_frequency_matrix_csv: str | Path | None = None,
    term_tfidf_matrix_csv: str | Path | None = None,
    evidence_cue_csv: str | Path | None = None,
    sentence_evidence_csv: str | Path | None = None,
    include_legacy_card: bool = False,
) -> dict[str, Any]:
    sentences = section_document(text)
    section_bow = SectionBowVocabulary.load(section_bow_csv)
    term_matrix = DocumentTermMatrix.load(term_frequency_matrix_csv, term_tfidf_matrix_csv, card.doc_id)
    cue_lexicon = EvidenceCueLexicon.load(evidence_cue_csv)
    external_evidence = ExternalEvidenceIndex.load(sentence_evidence_csv, card.doc_id)
    corpus_statistics = build_corpus_statistics(sentences)
    section_profiles = build_section_profiles(sentences, section_bow, term_matrix)
    enriched_units = [
        enrich_unit(unit, idx, sentences, section_bow, term_matrix, cue_lexicon, external_evidence)
        for idx, unit in enumerate(card.units, start=1)
    ]
    enriched_units.sort(key=lambda item: item["importance"]["downstream_relevance_score"], reverse=True)
    for idx, item in enumerate(enriched_units, start=1):
        item["unit_id"] = f"ecu_{idx:03d}"
        item["rank"] = idx

    payload: dict[str, Any] = {
        "schema_version": "evidence_units_v1",
        "document": {
            "doc_id": card.doc_id,
            "title": card.title,
            "source_path": str(Path(source_path).resolve()) if source_path else "",
            "sentence_count": len(sentences),
            "section_counts": dict(Counter(sentence.section for sentence in sentences)),
        },
        "pipeline": {
            "output_type": "evidence_grounded_concept_units_json",
            "keyword_checkpoint": str(keyword_checkpoint or ""),
            "structure_checkpoint": str(structured_checkpoint or ""),
            "legacy_bow_csv": str(bow_csv or ""),
            "section_bow_csv": str(section_bow_csv or ""),
            "term_frequency_matrix_csv": str(term_frequency_matrix_csv or ""),
            "term_tfidf_matrix_csv": str(term_tfidf_matrix_csv or ""),
            "evidence_cue_csv": str(evidence_cue_csv or ""),
            "sentence_evidence_csv": str(sentence_evidence_csv or ""),
            "card_markdown_generated": False,
            "overview_generated": False,
        },
        "course_nlp_features": course_feature_manifest(),
        "corpus_statistics": corpus_statistics,
        "document_term_matrix": document_term_matrix_summary(term_matrix),
        "section_profiles": section_profiles,
        "evidence_units": enriched_units,
        "external_evidence_candidates": external_evidence_summary(external_evidence),
        "concept_graph": build_concept_graph(enriched_units),
        "downstream_contract": {
            "intended_use": "Feed this JSON to a later local open-source model to write an overview.",
            "grounding_rule": "The downstream generator should write from evidence_units.location.sentence and context_window, not from unsupported outside knowledge.",
            "recommended_order": ["intro", "related_work", "method", "experiment", "conclusion"],
            "score_warning": "Scores are ranking and filtering signals, not calibrated probabilities.",
        },
    }
    if include_legacy_card:
        payload["legacy_paper_card"] = card.to_dict()
    return payload


def enrich_unit(
    unit: ConceptUnit,
    original_rank: int,
    sentences: list[SentenceRecord],
    section_bow: SectionBowVocabulary | None,
    term_matrix: DocumentTermMatrix | None,
    cue_lexicon: EvidenceCueLexicon | None,
    external_evidence: ExternalEvidenceIndex | None,
) -> dict[str, Any]:
    sentence = sentences[unit.sentence_index] if 0 <= unit.sentence_index < len(sentences) else None
    section = sentence.section if sentence else unit.section
    evidence_sentence = sentence.text if sentence else unit.evidence_sentence
    tokens = tokenize(unit.phrase)
    bow_match = section_bow.match_phrase(unit.phrase, section) if section_bow else None
    matrix_score = term_matrix.feature_score(section, unit.phrase) if term_matrix else None
    cue_hits = cue_lexicon.sentence_hits(evidence_sentence, section=section) if cue_lexicon else []
    external_overlap = external_evidence.sentence_overlap(evidence_sentence) if external_evidence else {"matched": False}
    bow_confidence = bow_match.confidence if bow_match else 0.0
    section_prior = bow_match.section_prior if bow_match else 0.0
    section_match_bonus = 0.07 if bow_match and bow_match.term.section == normalize_section(section) else 0.0
    role_prior_bonus = role_prior_bonus_for(unit.role, bow_match, cue_hits)
    cue_bonus = min(0.08, 0.035 * len(cue_hits))
    matrix_tfidf_score = normalized_score(matrix_score.tfidf, term_matrix.max_tfidf) if matrix_score and term_matrix else 0.0
    ngram_specificity = min(1.0, max(0.0, len(tokens) / 4.0))
    downstream_score = clamp(
        0.52 * unit.importance
        + 0.14 * unit.evidence_score
        + 0.08 * unit.role_score
        + 0.10 * bow_confidence
        + 0.05 * section_prior
        + 0.05 * matrix_tfidf_score
        + 0.04 * ngram_specificity
        + cue_bonus
        + role_prior_bonus
        + section_match_bonus
    )
    canonical = bow_match.term.canonical_term if bow_match else unit.phrase
    return {
        "unit_id": f"ecu_{original_rank:03d}",
        "rank": original_rank,
        "phrase": {
            "surface": unit.phrase,
            "canonical": canonical,
            "normalized": normalize_text(canonical),
            "tokens": tokens,
            "ngram_len": len(tokens),
            "source": phrase_sources(bow_match),
            "aliases": list(bow_match.term.aliases[:8]) if bow_match else [],
        },
        "location": {
            "section": section,
            "sentence_index": unit.sentence_index,
            "sentence": evidence_sentence,
            "context_window": context_window(sentences, unit.sentence_index),
        },
        "role": {
            "label": unit.role,
            "score": round_float(unit.role_score),
        },
        "evidence": {
            "score": round_float(unit.evidence_score),
            "support_type": cue_hits[0].evidence_type if cue_hits else ROLE_SUPPORT_TYPES.get(unit.role, "general_evidence"),
            "cue_hits": cue_metadata(cue_hits),
            "external_candidate_overlap": external_overlap,
        },
        "importance": {
            "original_unit_score": round_float(unit.importance),
            "downstream_relevance_score": round_float(downstream_score),
            "components": {
                "bio_boundary_score": round_float(unit.boundary_score),
                "sentence_evidence_score": round_float(unit.evidence_score),
                "sentence_role_score": round_float(unit.role_score),
                "bow_match_confidence": round_float(bow_confidence),
                "laplace_section_prior": round_float(section_prior),
                "document_tfidf_score": round_float(matrix_tfidf_score),
                "evidence_cue_bonus": round_float(cue_bonus),
                "role_prior_bonus": round_float(role_prior_bonus),
                "section_match_bonus": round_float(section_match_bonus),
                "ngram_specificity": round_float(ngram_specificity),
            },
        },
        "bow_metadata": bow_metadata(bow_match),
        "document_term_matrix": matrix_metadata(matrix_score, term_matrix),
        "course_feature_trace": {
            "tokenization": tokens,
            "normalization": normalize_text(unit.phrase),
            "ngram": f"{len(tokens)}-gram" if tokens else "0-gram",
            "bow_feature": bool(bow_match),
            "tfidf_feature": matrix_score is not None,
            "evidence_cue_feature": bool(cue_hits),
            "laplace_smoothed_section_prior": round_float(section_prior),
            "canonicalization": {
                "observed": unit.phrase,
                "canonical": canonical,
                "channel_score": round_float(bow_confidence),
            },
        },
        "selection_reason": selection_reasons(unit, bow_match, cue_hits, matrix_score, downstream_score),
    }


def build_corpus_statistics(sentences: list[SentenceRecord]) -> dict[str, Any]:
    tokens_by_section: dict[str, list[str]] = {section: [] for section in CANONICAL_SECTIONS}
    all_tokens: list[str] = []
    for sentence in sentences:
        sentence_tokens = tokenize(sentence.text)
        tokens_by_section.setdefault(sentence.section, []).extend(sentence_tokens)
        all_tokens.extend(sentence_tokens)
    content_tokens = [token for token in all_tokens if token not in STOPWORDS and len(token) > 1]
    type_count = len(set(all_tokens))
    token_count = len(all_tokens)
    return {
        "token_count": token_count,
        "type_count": type_count,
        "type_token_ratio": round_float(type_count / token_count if token_count else 0.0),
        "section_token_counts": {section: len(tokens) for section, tokens in tokens_by_section.items()},
        "top_unigrams": top_ngrams(content_tokens, 1, 20),
        "top_bigrams": top_ngrams(content_tokens, 2, 20),
        "top_trigrams": top_ngrams(content_tokens, 3, 20),
    }


def build_section_profiles(
    sentences: list[SentenceRecord],
    section_bow: SectionBowVocabulary | None,
    term_matrix: DocumentTermMatrix | None,
) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    for section in CANONICAL_SECTIONS:
        section_sentences = [sentence for sentence in sentences if sentence.section == section]
        tokens = [token for sentence in section_sentences for token in tokenize(sentence.text)]
        bow_counter: Counter[str] = Counter()
        bow_examples: dict[str, SectionBowMatch] = {}
        if section_bow:
            for sentence in section_sentences:
                for hit in section_bow.sentence_hits(sentence.text, section=section):
                    canonical = hit.term.canonical_term
                    bow_counter[canonical] += 1
                    current = bow_examples.get(canonical)
                    if current is None or hit.confidence > current.confidence:
                        bow_examples[canonical] = hit
        profiles[section] = {
            "sentence_count": len(section_sentences),
            "token_count": len(tokens),
            "bow_hit_count": sum(bow_counter.values()),
            "top_bow_terms": [
                {
                    "canonical_term": canonical,
                    "count": count,
                    "section_prior": round_float(bow_examples[canonical].section_prior),
                    "confidence_score": round_float(bow_examples[canonical].term.confidence_score),
                    "wiki_category": bow_examples[canonical].term.wiki_category,
                }
                for canonical, count in bow_counter.most_common(12)
            ],
            "top_tfidf_terms": [
                matrix_feature_to_dict(item)
                for item in (term_matrix.top_features(section=section, limit=12) if term_matrix else [])
            ],
        }
    return profiles


def build_concept_graph(units: list[dict[str, Any]]) -> dict[str, Any]:
    nodes = [
        {
            "id": unit["unit_id"],
            "canonical": unit["phrase"]["canonical"],
            "section": unit["location"]["section"],
            "role": unit["role"]["label"],
            "score": unit["importance"]["downstream_relevance_score"],
        }
        for unit in units
    ]
    edges: list[dict[str, Any]] = []
    by_sentence: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_section_role: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        by_sentence[int(unit["location"]["sentence_index"])].append(unit)
        by_section_role[(unit["location"]["section"], unit["role"]["label"])].append(unit)
    for group in by_sentence.values():
        edges.extend(pair_edges(group, "same_evidence_sentence", 0.9))
    for group in by_section_role.values():
        edges.extend(pair_edges(group[:4], "same_section_role", 0.55))
    return {"nodes": nodes, "edges": edges[:80]}


def pair_edges(group: list[dict[str, Any]], edge_type: str, weight: float) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for left_idx in range(len(group)):
        for right_idx in range(left_idx + 1, len(group)):
            edges.append(
                {
                    "source": group[left_idx]["unit_id"],
                    "target": group[right_idx]["unit_id"],
                    "type": edge_type,
                    "weight": weight,
                }
            )
    return edges


def course_feature_manifest() -> dict[str, Any]:
    return {
        "corpus_building": {
            "implemented_as": "The TXT paper is converted into a section-aware annotated mini-corpus.",
            "json_fields": ["document", "corpus_statistics", "section_profiles", "evidence_units"],
        },
        "tokenization": {
            "implemented_as": "Every concept phrase and sentence is normalized into tokens for counts and n-grams.",
            "json_fields": ["corpus_statistics.token_count", "evidence_units[].phrase.tokens"],
        },
        "normalization": {
            "implemented_as": "Lower-casing and punctuation cleanup are used for phrase matching and deduplication.",
            "json_fields": ["evidence_units[].phrase.normalized", "evidence_units[].course_feature_trace.normalization"],
        },
        "n_gram": {
            "implemented_as": "Concept length and document top unigrams/bigrams/trigrams are exposed.",
            "json_fields": ["corpus_statistics.top_unigrams", "corpus_statistics.top_bigrams", "corpus_statistics.top_trigrams", "evidence_units[].phrase.ngram_len"],
        },
        "bow_and_naive_bayes": {
            "implemented_as": "Section BoW hits act as class-conditional section evidence for intro/method/experiment/conclusion style decisions.",
            "json_fields": ["section_profiles[].top_bow_terms", "evidence_units[].bow_metadata"],
        },
        "tf_idf_document_term_matrix": {
            "implemented_as": "Document-level frequency and TF-IDF matrices expose which section concepts are characteristic of this paper.",
            "json_fields": ["document_term_matrix", "section_profiles[].top_tfidf_terms", "evidence_units[].document_term_matrix"],
        },
        "laplace_smoothing": {
            "implemented_as": "P(section | concept) uses add-one smoothing, so unseen section-concept pairs are not forced to zero.",
            "json_fields": ["evidence_units[].importance.components.laplace_section_prior"],
        },
        "evidence_cue_lexicon": {
            "implemented_as": "Cue phrases such as propose, achieve, outperform, and BLEU give transparent rule-based evidence hints.",
            "json_fields": ["evidence_units[].evidence.cue_hits", "external_evidence_candidates"],
        },
        "sequence_labeling": {
            "implemented_as": "The first model contributes BIO boundary evidence for concept phrase extraction.",
            "json_fields": ["evidence_units[].importance.components.bio_boundary_score"],
        },
        "canonicalization_noisy_channel_style": {
            "implemented_as": "Observed surface phrases are mapped to canonical terms with an alias confidence score.",
            "json_fields": ["evidence_units[].course_feature_trace.canonicalization"],
        },
    }


def bow_metadata(match: SectionBowMatch | None) -> dict[str, Any]:
    if match is None:
        return {"matched": False}
    term = match.term
    return {
        "matched": True,
        "matched_alias": match.alias,
        "match_quality": round_float(match.match_quality),
        "section": term.section,
        "original_section": term.original_section,
        "canonical_term": term.canonical_term,
        "display_term": term.display_term,
        "term_type": term.term_type,
        "role_prior": term.role_prior,
        "ngram_label": term.ngram_label,
        "pos_pattern": term.pos_pattern,
        "is_generic": term.is_generic,
        "source_corpus": term.source_corpus,
        "wiki_category": term.wiki_category,
        "wiki_url": term.wiki_url,
        "wikidata_id": term.wikidata_id,
        "source": term.source,
        "match_type": term.match_type,
        "notes": term.notes,
        "document_frequency": term.document_frequency,
        "total_frequency": term.total_frequency,
        "confidence_score": round_float(term.confidence_score),
    }


def cue_metadata(cues: list[EvidenceCue]) -> list[dict[str, Any]]:
    return [
        {
            "cue_phrase": cue.cue_phrase,
            "evidence_type": cue.evidence_type,
            "role_prior": cue.role_prior,
            "section_hint": cue.section_hint,
            "notes": cue.notes,
        }
        for cue in cues
    ]


def matrix_metadata(score: MatrixFeatureScore | None, matrix: DocumentTermMatrix | None) -> dict[str, Any]:
    if score is None:
        return {"matched": False}
    return {
        "matched": True,
        "feature_name": score.feature_name,
        "section": score.section,
        "term": score.term,
        "frequency": round_float(score.frequency),
        "tfidf": round_float(score.tfidf),
        "normalized_frequency": round_float(normalized_score(score.frequency, matrix.max_frequency if matrix else 0.0)),
        "normalized_tfidf": round_float(normalized_score(score.tfidf, matrix.max_tfidf if matrix else 0.0)),
    }


def document_term_matrix_summary(matrix: DocumentTermMatrix | None) -> dict[str, Any]:
    if matrix is None:
        return {"available": False}
    return {
        "available": True,
        "nonzero_frequency_features": sum(1 for value in matrix.frequency.values() if value > 0.0),
        "nonzero_tfidf_features": sum(1 for value in matrix.tfidf.values() if value > 0.0),
        "top_tfidf_features": [matrix_feature_to_dict(item) for item in matrix.top_features(limit=25)],
        "top_tfidf_features_by_section": {
            section: [matrix_feature_to_dict(item) for item in matrix.top_features(section=section, limit=12)]
            for section in CANONICAL_SECTIONS
        },
    }


def external_evidence_summary(index: ExternalEvidenceIndex | None) -> dict[str, Any]:
    if index is None:
        return {"available": False, "count": 0, "top_candidates": []}
    return {
        "available": True,
        "count": len(index.candidates),
        "top_candidates": index.top_candidates(limit=25),
    }


def matrix_feature_to_dict(item: MatrixFeatureScore) -> dict[str, Any]:
    return {
        "feature_name": item.feature_name,
        "section": item.section,
        "term": item.term,
        "frequency": round_float(item.frequency),
        "tfidf": round_float(item.tfidf),
    }


def role_prior_bonus_for(role: str, match: SectionBowMatch | None, cues: list[EvidenceCue]) -> float:
    bonus = 0.0
    if match and match.term.role_prior and normalize_text(match.term.role_prior) == normalize_text(role):
        bonus += 0.045
    if any(normalize_text(cue.role_prior) == normalize_text(role) for cue in cues if cue.role_prior):
        bonus += 0.035
    return min(0.08, bonus)


def selection_reasons(
    unit: ConceptUnit,
    match: SectionBowMatch | None,
    cue_hits: list[EvidenceCue],
    matrix_score: MatrixFeatureScore | None,
    downstream_score: float,
) -> list[str]:
    reasons = [
        f"BIO boundary retained phrase with score {round_float(unit.boundary_score)}",
        f"structure model assigned role '{unit.role}' with score {round_float(unit.role_score)}",
    ]
    if unit.evidence_score >= 0.45:
        reasons.append(f"evidence score {round_float(unit.evidence_score)} keeps the source sentence usable as grounding")
    if match is not None:
        reasons.append(
            "section BoW matched canonical term "
            f"'{match.term.canonical_term}' with confidence {round_float(match.confidence)}"
        )
        if match.term.role_prior:
            reasons.append(f"BoW role prior suggests '{match.term.role_prior}'")
    if matrix_score is not None:
        reasons.append(
            f"document-term matrix links the phrase to feature '{matrix_score.feature_name}' "
            f"with tf-idf {round_float(matrix_score.tfidf)}"
        )
    if cue_hits:
        reasons.append(
            "evidence cue lexicon matched "
            + ", ".join(cue.cue_phrase for cue in cue_hits[:3])
        )
    reasons.append(f"downstream relevance score is {round_float(downstream_score)}")
    return reasons


def context_window(sentences: list[SentenceRecord], index: int) -> dict[str, str]:
    return {
        "previous": sentences[index - 1].text if 0 < index < len(sentences) else "",
        "current": sentences[index].text if 0 <= index < len(sentences) else "",
        "next": sentences[index + 1].text if 0 <= index + 1 < len(sentences) else "",
    }


def top_ngrams(tokens: list[str], n: int, limit: int) -> list[dict[str, Any]]:
    if len(tokens) < n:
        return []
    counts = Counter(" ".join(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1))
    return [{"term": term, "count": count} for term, count in counts.most_common(limit)]


def tokenize(text: str) -> list[str]:
    return [token for token in normalize_text(text).split() if token]


def phrase_sources(match: SectionBowMatch | None) -> list[str]:
    sources = ["BIO"]
    if match is not None:
        sources.append("section_bow")
    return sources


def alias_match_quality(normalized_phrase: str, normalized_alias: str) -> float:
    if normalized_phrase == normalized_alias:
        return 1.0
    phrase_tokens = set(normalized_phrase.split())
    alias_tokens = set(normalized_alias.split())
    if not phrase_tokens or not alias_tokens:
        return 0.0
    if len(alias_tokens) >= 2 and alias_tokens.issubset(phrase_tokens):
        return 0.84
    if len(phrase_tokens) >= 2 and phrase_tokens.issubset(alias_tokens):
        return 0.78
    jaccard = token_jaccard(normalized_phrase, normalized_alias)
    return jaccard if jaccard >= 0.72 else 0.0


def collect_aliases(row: dict[str, str], canonical: str) -> list[str]:
    aliases: set[str] = {canonical}
    for field in ("display_term", "term", "normalized_term", "canonical_term"):
        value = (row.get(field) or "").strip()
        if value:
            aliases.add(value)
    for field in ("matched_surface_terms", "search_terms"):
        for part in (row.get(field) or "").split(";"):
            value = part.strip()
            if value:
                aliases.add(value)
    return sorted(aliases, key=lambda item: (len(item.split()), len(item)), reverse=True)


def load_matrix_row(path: str | Path | None, doc_id: str) -> dict[str, float]:
    if not path:
        return {}
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (row.get("document") or "").strip() != doc_id:
                continue
            return {
                key: safe_float(value, 0.0)
                for key, value in row.items()
                if key != "document" and safe_float(value, 0.0) > 0.0
            }
    return {}


def split_matrix_feature(feature: str) -> tuple[str, str]:
    if "__" not in feature:
        return "intro", feature
    section, term = feature.split("__", 1)
    return normalize_section(section), term


def infer_bow_confidence(row: dict[str, str], is_generic: bool) -> float:
    explicit = row.get("confidence_score")
    if explicit not in (None, ""):
        return safe_float(explicit, 0.0)
    confidence = 0.82
    if (row.get("source_corpus") or "").strip().lower() == "both":
        confidence += 0.06
    if (row.get("wikidata_id") or "").strip():
        confidence += 0.03
    if is_generic:
        confidence -= 0.20
    return max(0.35, min(0.95, confidence))


def normalize_section(section: str | None) -> str:
    if not section:
        return "intro"
    key = section.lower().replace("_", " ").strip()
    return SECTION_ALIASES.get(key, key if key in CANONICAL_SECTIONS else "intro")


def safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: object, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def normalized_score(value: float, maximum: float) -> float:
    if maximum <= 0.0:
        return 0.0
    return max(0.0, min(1.0, float(value) / maximum))


def round_float(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
