from __future__ import annotations

import math
import re

from ske.data.text_utils import normalize_text, token_jaccard

from .schema import SECTION_DEFAULT_ROLES


ROLE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("objective", re.compile(r"\b(we|this paper|our work)\s+(aim|aims|seek|seeks|focus|focuses|investigate|study|address|propose|introduce)", re.I)),
    ("problem", re.compile(r"\b(challenge|problem|difficulty|difficult|hard|limitation|bottleneck|lack|scarce|cannot|fail|fails)\b", re.I)),
    ("motivation", re.compile(r"\b(motivat|important|need|necessary|crucial|desirable)\b", re.I)),
    ("prior_work", re.compile(r"\b(previous|prior|existing|recent|earlier|baseline|state-of-the-art|sota)\b", re.I)),
    ("gap", re.compile(r"\b(however|nevertheless|still|remain|gap|open question|not yet|limited)\b", re.I)),
    ("comparison", re.compile(r"\b(compare|comparison|unlike|different from|outperform|better than)\b", re.I)),
    ("core_method", re.compile(r"\b(we propose|we introduce|our method|our model|we present|framework|architecture)\b", re.I)),
    ("component", re.compile(r"\b(module|component|layer|encoder|decoder|classifier|head|embedding|attention)\b", re.I)),
    ("mechanism", re.compile(r"\b(compute|learn|encode|decode|align|aggregate|select|score|optimize|generate)\b", re.I)),
    ("process", re.compile(r"\b(train|training|fine-tune|pretrain|pipeline|procedure|algorithm|step)\b", re.I)),
    ("dataset", re.compile(r"\b(dataset|corpus|benchmark|data set|training set|test set|validation set)\b", re.I)),
    ("metric", re.compile(r"\b(accuracy|precision|recall|f1|rouge|bleu|meteor|score|metric)\b", re.I)),
    ("baseline", re.compile(r"\b(baseline|ablated|ablation|compare with|compared with)\b", re.I)),
    ("result", re.compile(r"\b(result|achieve|outperform|improve|gain|performance|significant|state-of-the-art)\b", re.I)),
    ("ablation", re.compile(r"\b(ablation|remove|without|variant|analysis)\b", re.I)),
    ("contribution", re.compile(r"\b(contribution|contribute|we make|we show|we demonstrate)\b", re.I)),
    ("finding", re.compile(r"\b(find|finding|show|shows|suggest|indicate|conclude)\b", re.I)),
    ("future_work", re.compile(r"\b(future work|future|further|limitation|limitations)\b", re.I)),
]


SECTION_ROLE_PRIOR = {
    "intro": ["background", "problem", "motivation", "objective"],
    "related_work": ["prior_work", "limitation", "gap", "comparison"],
    "method": ["core_method", "component", "mechanism", "process"],
    "experiment": ["dataset", "metric", "baseline", "result", "ablation"],
    "conclusion": ["contribution", "finding", "limitation", "future_work"],
}


def infer_role_from_sentence(sentence: str, section: str) -> tuple[str, float]:
    section_roles = SECTION_ROLE_PRIOR.get(section, SECTION_DEFAULT_ROLES.get(section, ["background"]))
    for role, pattern in ROLE_PATTERNS:
        if role in section_roles and pattern.search(sentence):
            return role, 0.82
    for role, pattern in ROLE_PATTERNS:
        if pattern.search(sentence):
            return role, 0.66
    return section_roles[0], 0.45


def infer_importance(sentence: str, role: str, evidence_score: float = 0.0) -> float:
    """Legacy fallback for old card rendering; not used by Stage 2 training."""
    role_weight = {
        "objective": 0.74,
        "problem": 0.68,
        "gap": 0.66,
        "core_method": 0.78,
        "component": 0.64,
        "mechanism": 0.68,
        "dataset": 0.60,
        "metric": 0.56,
        "result": 0.80,
        "contribution": 0.76,
        "finding": 0.74,
        "limitation": 0.62,
        "future_work": 0.50,
    }.get(role, 0.45)
    cue_bonus = 0.08 if re.search(r"\b(we|our|propose|achieve|show|result|contribution)\b", sentence, re.I) else 0.0
    return max(0.0, min(1.0, role_weight + cue_bonus + 0.25 * evidence_score))


def best_summary_facet_role(sentence: str, summaries: dict[str, str]) -> tuple[str | None, float]:
    if not summaries:
        return None, 0.0
    facet_to_role = {"challenge": "problem", "approach": "core_method", "outcome": "result"}
    best_role = None
    best_score = 0.0
    for facet, summary in summaries.items():
        score = token_jaccard(sentence, summary)
        if score > best_score:
            best_role = facet_to_role.get(facet)
            best_score = score
    return best_role, best_score


def contains_evidence(sentence: str, evidence_strings: set[str]) -> bool:
    normalized_sentence = normalize_text(sentence)
    if not normalized_sentence:
        return False
    for evidence in evidence_strings:
        normalized_evidence = normalize_text(evidence)
        if normalized_evidence and (normalized_evidence in normalized_sentence or token_jaccard(normalized_sentence, normalized_evidence) >= 0.62):
            return True
    return False


def greedy_rouge_oracle_labels(
    sentences: list[str],
    summaries: dict[str, str],
    max_sentences_per_facet: int = 3,
    summary_tokens_per_sentence: int = 32,
) -> list[float]:
    """Build extractive oracle labels from facet summaries with greedy ROUGE-L gain."""
    labels = [0.0 for _ in sentences]
    if not sentences:
        return labels
    clean_sentences = [sentence.strip() for sentence in sentences]
    for summary in summaries.values():
        reference = summary.strip()
        reference_tokens = rouge_tokens(reference)
        if not reference_tokens:
            continue
        budget = min(max_sentences_per_facet, max(1, math.ceil(len(reference_tokens) / summary_tokens_per_sentence)))
        selected: list[int] = []
        remaining = set(range(len(clean_sentences)))
        current_score = 0.0
        for _ in range(budget):
            best_idx: int | None = None
            best_score = current_score
            for idx in remaining:
                candidate_indices = sorted([*selected, idx])
                candidate = " ".join(clean_sentences[item] for item in candidate_indices)
                score = rouge_l_f1(candidate, reference)
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is None:
                break
            labels[best_idx] = 1.0
            selected.append(best_idx)
            remaining.remove(best_idx)
            current_score = best_score
    return labels


def rouge_l_f1(candidate: str, reference: str) -> float:
    candidate_tokens = rouge_tokens(candidate)
    reference_tokens = rouge_tokens(reference)
    if not candidate_tokens or not reference_tokens:
        return 0.0
    lcs = lcs_length(candidate_tokens, reference_tokens)
    precision = lcs / max(len(candidate_tokens), 1)
    recall = lcs / max(len(reference_tokens), 1)
    if precision + recall <= 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize_text(text).lower())


def lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for idx, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[idx - 1] + 1)
            else:
                current.append(max(previous[idx], current[-1]))
        previous = current
    return previous[-1]
