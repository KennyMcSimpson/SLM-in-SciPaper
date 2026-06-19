from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .text_utils import find_normalized_phrase_spans, normalize_text


@dataclass(frozen=True)
class BowTerm:
    display_term: str
    canonical_term: str
    aliases: tuple[str, ...]
    confidence: float
    document_frequency: int
    total_frequency: int
    category: str


class BowVocabulary:
    def __init__(self, terms: list[BowTerm]) -> None:
        self.terms = terms
        self.alias_to_term: dict[str, BowTerm] = {}
        for term in terms:
            for alias in term.aliases:
                normalized = normalize_text(alias)
                if not normalized:
                    continue
                current = self.alias_to_term.get(normalized)
                if current is None or term.confidence > current.confidence:
                    self.alias_to_term[normalized] = term

    @classmethod
    def load(cls, path: str | Path | None) -> "BowVocabulary | None":
        if not path:
            return None
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        terms: list[BowTerm] = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                aliases = set()
                for field in ("display_term", "term", "normalized_term", "canonical_term"):
                    value = (row.get(field) or "").strip()
                    if value:
                        aliases.add(value)
                for field in ("matched_surface_terms", "search_terms"):
                    aliases.update(part.strip() for part in (row.get(field) or "").split(";") if part.strip())
                terms.append(
                    BowTerm(
                        display_term=(row.get("display_term") or "").strip(),
                        canonical_term=(row.get("canonical_term") or row.get("display_term") or "").strip(),
                        aliases=tuple(sorted(aliases, key=len, reverse=True)),
                        confidence=_safe_float(row.get("confidence_score"), 0.0),
                        document_frequency=_safe_int(row.get("document_frequency"), 0),
                        total_frequency=_safe_int(row.get("total_frequency"), 0),
                        category=(row.get("wiki_category") or "").strip(),
                    )
                )
        return cls(terms)

    def canonicalize(self, phrase: str) -> tuple[str, float]:
        term = self.alias_to_term.get(normalize_text(phrase))
        if term is None:
            return phrase, 0.0
        return term.canonical_term or phrase, term.confidence

    def sentence_hits(self, sentence: str, min_confidence: float = 0.7) -> list[BowTerm]:
        hits: list[BowTerm] = []
        seen: set[str] = set()
        for alias, term in self.alias_to_term.items():
            if term.confidence < min_confidence or term.canonical_term in seen:
                continue
            if find_normalized_phrase_spans(sentence, alias):
                hits.append(term)
                seen.add(term.canonical_term)
        return hits


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default

