from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


SECTION_ORDER = ["intro", "related_work", "method", "experiment", "conclusion"]

ROLE_PRIORITIES = {
    "intro": {"problem": 4, "objective": 4, "motivation": 3, "background": 2},
    "related_work": {"gap": 4, "limitation": 4, "comparison": 3, "prior_work": 2},
    "method": {"core_method": 5, "mechanism": 4, "component": 3, "process": 3},
    "experiment": {"result": 5, "metric": 4, "dataset": 3, "baseline": 3, "ablation": 3},
    "conclusion": {"contribution": 5, "finding": 4, "limitation": 3, "future_work": 3},
}

COMPATIBLE_ROLES = {
    "intro": {"problem", "objective", "motivation", "background"},
    "related_work": {"gap", "limitation", "comparison", "prior_work"},
    "method": {"core_method", "mechanism", "component", "process"},
    "experiment": {"result", "metric", "dataset", "baseline", "ablation"},
    "conclusion": {"contribution", "finding", "limitation", "future_work"},
}

GENERIC_TERMS = {
    "model",
    "models",
    "method",
    "methods",
    "approach",
    "system",
    "task",
    "paper",
    "work",
    "result",
    "results",
    "performance",
    "training",
    "dataset",
    "baseline",
    "experiment",
    "experiments",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an abstract-like overview from Evidence Units JSON.")
    parser.add_argument("--input_json", required=True, help="Evidence Units JSON produced by infer_paper_card.py.")
    parser.add_argument("--output_txt", default=None, help="Optional path for the generated overview text.")
    parser.add_argument("--output_json", default=None, help="Optional path for overview plus selected support fields.")
    parser.add_argument("--max_terms_per_section", type=int, default=4)
    parser.add_argument("--style", choices=["abstract", "plain"], default="abstract")
    parser.add_argument("--include_support_sentence", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_json)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    result = build_overview_payload(
        payload,
        max_terms_per_section=args.max_terms_per_section,
        style=args.style,
        include_support_sentence=args.include_support_sentence,
    )
    overview = result["overview"]

    if args.output_txt:
        output_txt = Path(args.output_txt)
        output_txt.parent.mkdir(parents=True, exist_ok=True)
        output_txt.write_text(overview + "\n", encoding="utf-8")
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(overview)


def build_overview_payload(
    payload: dict[str, Any],
    max_terms_per_section: int = 4,
    style: str = "abstract",
    include_support_sentence: bool = False,
) -> dict[str, Any]:
    document = payload.get("document", {})
    title = clean_title(document.get("title") or document.get("doc_id") or "the paper")
    units = payload.get("evidence_units", [])
    grouped = group_units_by_section(units)
    section_terms = {
        section: select_section_terms(grouped.get(section, []), section, max_terms_per_section)
        for section in SECTION_ORDER
    }
    evidence_snippets = {
        section: select_evidence_snippets(grouped.get(section, []), limit=2)
        for section in SECTION_ORDER
    }
    overview = build_abstract_text(title, section_terms, evidence_snippets, style=style, include_support_sentence=include_support_sentence)
    return {
        "document": document,
        "source_json": payload.get("document", {}).get("source_path", ""),
        "overview_type": "deterministic_evidence_grounded_overview",
        "overview": overview,
        "selected_terms": section_terms,
        "selected_evidence_snippets": evidence_snippets,
        "model_note": (
            "No external LLM is used. This is a deterministic evidence-grounded generator "
            "over the JSON fields produced by Stage1-Stage3."
        ),
    }


def group_units_by_section(units: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        section = unit.get("location", {}).get("section") or "intro"
        grouped[section].append(unit)
    for section, items in grouped.items():
        items.sort(key=lambda item: unit_score(item, section), reverse=True)
    return grouped


def select_section_terms(units: list[dict[str, Any]], section: str, limit: int) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for unit in sorted(units, key=lambda item: unit_score(item, section), reverse=True):
        role = unit.get("role", {}).get("label") or ""
        compatible = COMPATIBLE_ROLES.get(section)
        if compatible and role not in compatible:
            continue
        phrase = normalize_phrase(unit.get("phrase", {}).get("canonical") or "")
        if not useful_phrase(phrase):
            continue
        key = phrase.lower()
        if key in seen or any(key in old or old in key for old in seen):
            continue
        seen.add(key)
        terms.append(phrase)
        if len(terms) >= limit:
            break
    return terms


def select_evidence_snippets(units: list[dict[str, Any]], limit: int = 2) -> list[str]:
    snippets: list[str] = []
    seen: set[str] = set()
    for unit in units:
        sentence = clean_sentence(unit.get("location", {}).get("sentence") or "")
        if not sentence:
            continue
        key = normalize_for_match(sentence)
        if key in seen or any(key in old or old in key for old in seen):
            continue
        seen.add(key)
        snippets.append(trim_sentence(sentence, max_words=26))
        if len(snippets) >= limit:
            break
    return snippets


def build_abstract_text(
    title: str,
    section_terms: dict[str, list[str]],
    evidence_snippets: dict[str, list[str]],
    style: str,
    include_support_sentence: bool,
) -> str:
    intro = join_terms(section_terms.get("intro", []))
    related = join_terms(section_terms.get("related_work", []))
    method = join_terms(section_terms.get("method", []))
    experiment = join_terms(section_terms.get("experiment", []))
    conclusion = join_terms(section_terms.get("conclusion", []))

    if style == "plain":
        return build_plain_text(title, intro, related, method, experiment, conclusion)

    parts: list[str] = []
    if intro:
        parts.append(f"This paper addresses {intro}, framing the work around a concrete scientific or technical problem.")
    else:
        parts.append(f"This paper presents a research contribution in the area suggested by {title}.")
    if related:
        parts.append(f"Compared with prior work, the extracted evidence highlights {related} as the main context or gap.")
    if method:
        parts.append(f"The proposed approach is centered on {method}, which forms the main mechanism described in the paper.")
    if experiment:
        parts.append(f"The evaluation evidence focuses on {experiment}, linking the method to measurable empirical support.")
    if conclusion:
        parts.append(f"Overall, the paper's extracted conclusion emphasizes {conclusion}.")
    else:
        method_or_exp = method or experiment or intro
        if method_or_exp:
            parts.append(f"Overall, the evidence units connect the problem setting, the method design, and the evaluation around {method_or_exp}.")
    support = best_support_sentence(evidence_snippets) if include_support_sentence else ""
    if support:
        parts.append(f"The most direct extracted support states that {support}")
    return " ".join(parts)


def build_plain_text(title: str, intro: str, related: str, method: str, experiment: str, conclusion: str) -> str:
    clauses = [f"Paper: {title}."]
    if intro:
        clauses.append(f"Problem/background: {intro}.")
    if related:
        clauses.append(f"Related-work context: {related}.")
    if method:
        clauses.append(f"Method: {method}.")
    if experiment:
        clauses.append(f"Evidence/evaluation: {experiment}.")
    if conclusion:
        clauses.append(f"Conclusion: {conclusion}.")
    return " ".join(clauses)


def best_support_sentence(evidence_snippets: dict[str, list[str]]) -> str:
    for section in ("method", "experiment", "intro", "conclusion", "related_work"):
        snippets = evidence_snippets.get(section) or []
        if snippets:
            sentence = snippets[0].strip()
            if sentence and not sentence.endswith("."):
                sentence += "."
            return sentence
    return ""


def unit_score(unit: dict[str, Any], section: str) -> float:
    importance = float(unit.get("importance", {}).get("downstream_relevance_score") or 0.0)
    evidence = float(unit.get("evidence", {}).get("score") or 0.0)
    role = unit.get("role", {}).get("label") or ""
    role_bonus = ROLE_PRIORITIES.get(section, {}).get(role, 0) * 0.035
    return importance + 0.25 * evidence + role_bonus


def clean_title(title: str) -> str:
    title = re.sub(r"^\d{4}_", "", title)
    title = title.replace("_", " ")
    title = re.sub(r"\s+", " ", title).strip()
    return title or "the paper"


def normalize_phrase(phrase: str) -> str:
    phrase = re.sub(r"\s+", " ", phrase).strip(" ,.;:")
    return phrase


def useful_phrase(phrase: str) -> bool:
    if not phrase:
        return False
    lowered = phrase.lower()
    if lowered in GENERIC_TERMS:
        return False
    tokens = lowered.split()
    if len(tokens) == 1 and len(tokens[0]) <= 3:
        return False
    if any(token in {"eos", "pad", "cls", "sep"} for token in tokens):
        return False
    return True


def clean_sentence(sentence: str) -> str:
    sentence = re.sub(r"\s+", " ", sentence).strip(" ,;:")
    return sentence


def trim_sentence(sentence: str, max_words: int) -> str:
    words = sentence.split()
    if len(words) <= max_words:
        return sentence
    return " ".join(words[:max_words]).rstrip(" ,;:") + " ..."


def normalize_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def join_terms(terms: list[str]) -> str:
    if not terms:
        return ""
    if len(terms) == 1:
        return terms[0]
    if len(terms) == 2:
        return f"{terms[0]} and {terms[1]}"
    return ", ".join(terms[:-1]) + f", and {terms[-1]}"


if __name__ == "__main__":
    main()
