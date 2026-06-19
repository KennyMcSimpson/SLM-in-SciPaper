from __future__ import annotations

import re
from collections import Counter, defaultdict

from ske.data.text_utils import normalize_text

from .schema import CANONICAL_SECTIONS, ConceptUnit, PaperCard


SECTION_TITLES = {
    "intro": "背景与问题",
    "related_work": "相关工作与缺口",
    "method": "方法机制",
    "experiment": "实验与证据",
    "conclusion": "结论与局限",
}

ROLE_NAMES = {
    "background": "背景",
    "problem": "问题",
    "motivation": "动机",
    "objective": "目标",
    "prior_work": "已有工作",
    "limitation": "局限",
    "gap": "缺口",
    "comparison": "比较",
    "core_method": "核心方法",
    "component": "组件",
    "mechanism": "机制",
    "process": "流程",
    "dataset": "数据集",
    "metric": "指标",
    "baseline": "基线",
    "result": "结果",
    "ablation": "消融",
    "contribution": "贡献",
    "finding": "发现",
    "future_work": "未来工作",
}

SECTION_SUMMARY_ROLES = {
    "intro": {"background", "problem", "motivation", "objective"},
    "related_work": {"prior_work", "limitation", "gap", "comparison"},
    "method": {"core_method", "component", "mechanism", "process"},
    "experiment": {"dataset", "metric", "baseline", "result", "ablation"},
    "conclusion": {"contribution", "finding", "limitation", "future_work"},
}

SECTION_FOCUS_DEFAULTS = {
    "intro": "论文要解决的核心问题",
    "related_work": "已有方法及其不足",
    "method": "模型结构和关键机制",
    "experiment": "数据、指标和结果证据",
    "conclusion": "主要贡献、发现和后续限制",
}

LOW_VALUE_SUMMARY_PHRASES = {
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
}

SPECIAL_FOCUS_TERMS = {
    "autoregressive decoder",
    "bidirectional encoder",
    "bidirectional transformer encoder",
    "denoising autoencoder",
    "dense passage retriever",
    "dual-encoder architecture",
    "identity mapping",
    "inner product search",
    "masked language model",
    "multi-head attention",
    "noising function",
    "positional encoding",
    "residual function",
    "sequence-to-sequence model",
    "sequence-to-sequence transformer architecture",
    "shortcut connection",
    "self-attention",
    "target-side language model",
    "text infilling",
}

TERM_STOPWORDS = {
    "this",
    "that",
    "with",
    "from",
    "into",
    "using",
    "used",
    "uses",
    "their",
    "there",
    "these",
    "those",
    "which",
    "where",
    "while",
    "about",
    "after",
    "before",
    "between",
    "model",
    "models",
    "method",
    "methods",
    "paper",
    "work",
    "section",
    "table",
    "figure",
    "result",
    "results",
    "performance",
    "setup",
    "designed",
    "based",
    "validation",
    "overall",
    "original",
    "scores",
    "determined",
    "derived",
    "document",
    "combining",
    "corrupted",
    "multiple",
    "choice",
    "response",
    "publicly",
    "methodologies",
    "estimate",
    "report",
    "final",
    "hidden",
    "implemented",
    "standard",
    "state",
    "held-out",
}


def build_structured_summary(card: PaperCard, max_units_per_section: int = 5) -> dict[str, str]:
    grouped: dict[str, list[ConceptUnit]] = defaultdict(list)
    for unit in sorted(card.units, key=lambda item: item.importance, reverse=True):
        grouped[unit.section].append(unit)
    summary: dict[str, str] = {}
    for section in CANONICAL_SECTIONS:
        units = grouped.get(section, [])[:max_units_per_section]
        notes = card.section_notes.get(section, [])[:3]
        summary[section] = verbalize_section(section, units, notes)
    return summary


def build_one_paragraph_overview(summary: dict[str, str]) -> str:
    intro = _extract_focus(summary.get("intro", ""))
    related = _extract_focus(summary.get("related_work", ""))
    method = _extract_focus(summary.get("method", ""))
    experiment = _extract_focus(summary.get("experiment", ""))
    conclusion = _extract_focus(summary.get("conclusion", ""))

    parts: list[str] = []
    if intro:
        parts.append(f"这篇论文主要围绕{intro}展开，目标是回应相关任务中已有建模路线的局限。")
    else:
        parts.append("这篇论文围绕一个科学问题展开，但当前输入中背景证据不够稳定。")
    if related:
        parts.append(f"从相关工作看，已有研究主要集中在{related}，但仍然留下了本文希望进一步解决的缺口。")
    if method:
        parts.append(f"方法上，论文以{method}为核心，说明模型结构或算法机制如何支撑这一目标。")
    if experiment:
        parts.append(f"实验部分围绕{experiment}给出验证，展示方法在数据集、指标或对比结果上的表现。")
    if conclusion:
        parts.append(f"总体来看，论文的结论集中在{conclusion}，并把贡献、发现或后续限制收束到这一主线上。")
    else:
        parts.append("由于结论证据不够稳定，这里不强行补写额外结论。")
    return "".join(parts)


def _extract_focus(text: str) -> str:
    if not text or "当前没有足够稳定" in text:
        return ""
    match = re.search(r"“([^”]+)”", text)
    if match:
        return match.group(1).strip()
    cleaned = re.sub(r"可追溯证据：.*$", "", text).strip()
    cleaned = re.sub(r"^(这一部分|相关工作部分|方法部分|实验部分|结论部分)[^。]*。", "", cleaned).strip()
    return cleaned[:80].strip("，。； ")


def verbalize_section(section: str, units: list[ConceptUnit], notes: list[str]) -> str:
    evidence = merge_evidence(notes, [unit.evidence_sentence for unit in units if unit.evidence_sentence])
    if not units and not evidence:
        return fallback_sentence(section)

    phrases = unique_phrases(units)
    if section in {"method", "experiment"} and 0 < len(phrases) < 3:
        phrases = merge_focus_terms(phrases, infer_focus_terms_from_notes(evidence, max_terms=3))
    focus = join_items(phrases[:5])
    if not focus and units:
        focus = infer_focus_from_notes(evidence)
    if not focus:
        focus = SECTION_FOCUS_DEFAULTS.get(section, "论文内容")
    role_text = describe_roles(section, units)
    evidence_text = format_evidence(evidence[:2], phrases[:5])

    if section == "intro":
        return (
            f"这一部分把论文的出发点落在“{focus}”上。"
            f"{role_text}从证据看，作者先说明任务背景和已有路线的限制，再引出为什么需要新的建模方式。"
            f"可追溯证据：{evidence_text}"
        )
    if section == "related_work":
        return (
            f"相关工作部分主要围绕“{focus}”展开。"
            f"{role_text}它的作用不是重复介绍本文模型，而是说明已有方法提供了什么基础、还留下什么缺口。"
            f"可追溯证据：{evidence_text}"
        )
    if section == "method":
        return (
            f"方法部分的核心是“{focus}”。"
            f"{role_text}这些概念共同说明模型或算法怎样搭起来：主干结构是什么，关键组件怎样计算，最后怎样服务于论文目标。"
            f"可追溯证据：{evidence_text}"
        )
    if section == "experiment":
        return (
            f"实验部分主要落在“{focus}”。"
            f"{role_text}这一段卡片关心论文怎样证明方法有效：用了什么数据和指标，和谁比较，结果是否支撑主张。"
            f"可追溯证据：{evidence_text}"
        )
    if section == "conclusion":
        return (
            f"结论部分强调“{focus}”。"
            f"{role_text}这里适合作为整篇论文的收束：作者回到主要贡献和发现，同时给出仍然需要谨慎看待的限制或后续方向。"
            f"可追溯证据：{evidence_text}"
        )
    return f"{SECTION_TITLES.get(section, section)}：{focus}。证据：{evidence_text}"


def merge_evidence(notes: list[str], unit_sentences: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for sentence in notes + unit_sentences:
        key = normalize_text(sentence)
        if not key or key in seen:
            continue
        if any(key in old or old in key for old in seen):
            continue
        seen.add(key)
        merged.append(sentence)
    return merged


def unique_phrases(units: list[ConceptUnit]) -> list[str]:
    seen: set[str] = set()
    phrases: list[str] = []
    for unit in units:
        key = normalize_text(unit.phrase)
        if not key or key in seen or key in LOW_VALUE_SUMMARY_PHRASES:
            continue
        phrases.append(unit.phrase)
        seen.add(key)
    return phrases


def infer_focus_from_notes(sentences: list[str], max_terms: int = 3) -> str:
    return join_items(infer_focus_terms_from_notes(sentences, max_terms=max_terms))


def infer_focus_terms_from_notes(sentences: list[str], max_terms: int = 3) -> list[str]:
    counter: Counter[str] = Counter()
    for sentence in sentences:
        normalized_sentence = normalize_text(sentence)
        for term in SPECIAL_FOCUS_TERMS:
            if term in normalized_sentence:
                counter[term] += 8
        tokens = [
            token
            for token in re.findall(r"[a-z][a-z0-9-]+", sentence.lower())
            if len(token) >= 4 and token not in TERM_STOPWORDS and not token.isdigit()
        ]
        for ngram_size in (4, 3, 2):
            for idx in range(0, max(0, len(tokens) - ngram_size + 1)):
                phrase = " ".join(tokens[idx : idx + ngram_size])
                if not is_focus_candidate(phrase):
                    continue
                counter[phrase] += focus_candidate_score(phrase)
    return [term for term, _ in counter.most_common(max_terms)]


def is_focus_candidate(phrase: str) -> bool:
    if phrase in LOW_VALUE_SUMMARY_PHRASES:
        return False
    words = phrase.split()
    if len(words) < 2:
        return False
    if words[0] in TERM_STOPWORDS or words[-1] in TERM_STOPWORDS:
        return False
    head_words = {
        "architecture",
        "attention",
        "autoencoder",
        "autoencoders",
        "baseline",
        "benchmark",
        "dataset",
        "decoder",
        "encoder",
        "framework",
        "function",
        "mapping",
        "mechanism",
        "model",
        "objective",
        "pretraining",
        "retriever",
        "scheme",
        "shortcut",
        "training",
        "transformer",
    }
    if words[-1] in head_words:
        return True
    return any("-" in word for word in words)


def focus_candidate_score(phrase: str) -> int:
    words = phrase.split()
    score = 1
    if words[-1] in {"model", "encoder", "decoder", "architecture", "framework", "objective"}:
        score += 2
    if any("-" in word for word in words):
        score += 1
    return score


def merge_focus_terms(base_terms: list[str], fallback_terms: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for term in base_terms + fallback_terms:
        key = normalize_text(term)
        if not key or key in seen:
            continue
        replaced = False
        for idx, old_term in enumerate(list(merged)):
            old_key = normalize_text(old_term)
            if old_key and old_key in key and len(key.split()) > len(old_key.split()):
                seen.discard(old_key)
                merged[idx] = term
                seen.add(key)
                replaced = True
                break
        if replaced:
            continue
        if any(key in old for old in seen):
            continue
        seen.add(key)
        merged.append(term)
    return merged


def join_items(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return "、".join(items[:-1]) + "和" + items[-1]


def describe_roles(section: str, units: list[ConceptUnit]) -> str:
    if not units:
        return ""
    role_counts: dict[str, int] = defaultdict(int)
    allowed_roles = SECTION_SUMMARY_ROLES.get(section, set())
    for unit in units:
        if allowed_roles and unit.role not in allowed_roles:
            continue
        role_counts[unit.role] += 1
    ranked = sorted(role_counts.items(), key=lambda item: item[1], reverse=True)[:3]
    labels = [ROLE_NAMES.get(role, role) for role, _ in ranked if role != "none"]
    if not labels:
        return ""
    return f"角色上更偏向{join_items(labels)}。"


def format_evidence(sentences: list[str], focus_terms: list[str] | None = None) -> str:
    if not sentences:
        return "当前没有稳定证据句，暂不强行扩写。"
    return "；".join(trim_sentence(sentence, focus_terms=focus_terms) for sentence in sentences)


def trim_sentence(sentence: str, max_words: int = 30, focus_terms: list[str] | None = None) -> str:
    words = _display_words(sentence)
    if len(words) <= max_words:
        return " ".join(words)
    start = _find_focus_start(words, focus_terms or [])
    if start is None:
        start = _choose_clean_start(words)
    end = min(len(words), start + max_words)
    if end - start < max_words and end == len(words):
        start = max(0, end - max_words)
    return ("... " if start > 0 else "") + " ".join(words[start:end]) + (" ..." if end < len(words) else "")


def _display_words(sentence: str) -> list[str]:
    text = re.sub(r"\s+", " ", sentence.replace("|", " ")).strip()
    return text.split()


def _find_focus_start(words: list[str], focus_terms: list[str]) -> int | None:
    lowered = [word.lower().strip(".,;:()[]{}") for word in words]
    for term in focus_terms:
        term_words = [word for word in normalize_text(term).split() if word]
        if not term_words:
            continue
        size = len(term_words)
        for idx in range(0, max(0, len(lowered) - size + 1)):
            if lowered[idx : idx + size] == term_words:
                return max(0, idx - 8)
    return None


def _choose_clean_start(words: list[str]) -> int:
    lowered = [word.lower().strip(".,;:()[]{}") for word in words]
    cue_patterns = [
        ("in", "this", "paper"),
        ("in", "this", "work"),
        ("we", "propose"),
        ("we", "introduce"),
        ("we", "present"),
        ("we", "show"),
        ("we", "train"),
        ("we", "use"),
        ("our", "model"),
        ("results", "show"),
        ("this", "section"),
    ]
    for pattern in cue_patterns:
        size = len(pattern)
        for idx in range(0, max(0, len(lowered) - size + 1)):
            if tuple(lowered[idx : idx + size]) == pattern:
                return max(0, idx - 2)
    for idx, word in enumerate(lowered[:40]):
        if _looks_like_formula_token(word):
            continue
        if len(word) >= 4:
            return idx
    return 0


def _looks_like_formula_token(token: str) -> bool:
    if not token or token.isdigit():
        return True
    if len(token) == 1 and token not in {"a", "i"}:
        return True
    return bool(re.fullmatch(r"[a-z]\d+", token))


def fallback_sentence(section: str) -> str:
    return {
        "intro": "当前没有足够稳定的背景/问题证据，暂不强行概括。",
        "related_work": "当前没有足够稳定的相关工作/缺口证据，暂不强行概括。",
        "method": "当前没有足够稳定的方法机制证据，暂不强行概括。",
        "experiment": "当前没有足够稳定的实验结果证据，暂不强行概括。",
        "conclusion": "当前没有足够稳定的结论/局限证据，暂不强行概括。",
    }.get(section, "当前没有足够稳定的证据。")


def render_card_markdown(card: PaperCard) -> str:
    lines = [f"# Paper Card: {card.title or card.doc_id}", ""]
    lines.append("## Evidence-grounded Concept Units")
    lines.append("")
    lines.append("| section | phrase | role | importance | evidence |")
    lines.append("|---|---|---|---:|---|")
    for unit in sorted(card.units, key=lambda item: item.importance, reverse=True):
        evidence = trim_sentence(unit.evidence_sentence.replace("|", " "), max_words=42, focus_terms=[unit.phrase])
        lines.append(f"| {unit.section} | {unit.phrase} | {unit.role} | {unit.importance:.3f} | {evidence} |")
    lines.append("")
    lines.append("## Section Evidence Notes")
    lines.append("")
    for section in CANONICAL_SECTIONS:
        notes = card.section_notes.get(section, [])
        if not notes:
            continue
        lines.append(f"### {SECTION_TITLES[section]}")
        lines.append("")
        for note in notes:
            lines.append(f"- {trim_sentence(note, max_words=42)}")
        lines.append("")
    lines.append("## One-paragraph Paper Overview")
    lines.append("")
    lines.append(build_one_paragraph_overview(card.summary))
    lines.append("")
    lines.append("## Five-part Structured Summary")
    lines.append("")
    for section in CANONICAL_SECTIONS:
        lines.append(f"### {SECTION_TITLES[section]}")
        lines.append("")
        lines.append(card.summary.get(section, fallback_sentence(section)))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
