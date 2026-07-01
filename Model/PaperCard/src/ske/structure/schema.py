from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


CANONICAL_SECTIONS = ["intro", "related_work", "method", "experiment", "conclusion"]
SECTION_TO_ID = {name: idx for idx, name in enumerate(CANONICAL_SECTIONS)}
ID_TO_SECTION = {idx: name for name, idx in SECTION_TO_ID.items()}

ROLE_LABELS = [
    "none",
    "background",
    "problem",
    "motivation",
    "objective",
    "prior_work",
    "limitation",
    "gap",
    "comparison",
    "core_method",
    "component",
    "mechanism",
    "process",
    "dataset",
    "metric",
    "baseline",
    "result",
    "ablation",
    "contribution",
    "finding",
    "future_work",
]
ROLE_TO_ID = {name: idx for idx, name in enumerate(ROLE_LABELS)}
ID_TO_ROLE = {idx: name for name, idx in ROLE_TO_ID.items()}

SECTION_DEFAULT_ROLES = {
    "intro": ["background", "problem", "motivation", "objective"],
    "related_work": ["prior_work", "limitation", "gap", "comparison"],
    "method": ["core_method", "component", "mechanism", "process"],
    "experiment": ["dataset", "metric", "baseline", "result", "ablation"],
    "conclusion": ["contribution", "finding", "limitation", "future_work"],
}


@dataclass
class SentenceRecord:
    text: str
    section: str = "intro"
    role: str | None = None
    evidence_label: float | None = None
    importance_label: float | None = None
    section_title: str = ""
    source: str = ""
    sentence_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StructuredDocument:
    doc_id: str
    title: str = ""
    abstract: str = ""
    sentences: list[SentenceRecord] = field(default_factory=list)
    summaries: dict[str, str] = field(default_factory=dict)
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sentences"] = [sentence.to_dict() for sentence in self.sentences]
        return payload


@dataclass
class ConceptUnit:
    section: str
    phrase: str
    role: str
    evidence_sentence: str
    importance: float
    sentence_index: int
    s_boundary: float = 0.0
    s_selector: float = 0.0
    s_bow: float = 0.0
    s_candidate: float = 0.0
    s_coverage: float = 0.0
    s_rerank: float = 0.0
    boundary_score: float = 0.0
    evidence_score: float = 0.0
    sentence_importance_score: float = 0.0
    role_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PaperCard:
    doc_id: str
    title: str = ""
    units: list[ConceptUnit] = field(default_factory=list)
    section_notes: dict[str, list[str]] = field(default_factory=dict)
    summary: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "units": [unit.to_dict() for unit in self.units],
            "section_notes": self.section_notes,
            "summary": self.summary,
        }


def normalize_section_name(section: str | None) -> str:
    if not section:
        return "intro"
    if section in SECTION_TO_ID:
        return section
    lowered = section.lower().strip()
    if lowered in SECTION_TO_ID:
        return lowered
    return "intro"


def normalize_role_name(role: str | None) -> str:
    if not role:
        return "none"
    lowered = role.lower().strip()
    return lowered if lowered in ROLE_TO_ID else "none"
