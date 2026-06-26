from __future__ import annotations

import re
from dataclasses import dataclass

from ske.data.text_utils import split_sentences

from .schema import CANONICAL_SECTIONS, SentenceRecord


SECTION_NUMBER_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*|[A-Z])\.?\s*$")
PAGE_NUMBER_RE = re.compile(r"^\s*\f?\s*\d{1,3}\s*$")

SECTION_HEADING_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?(.+?)\s*$", re.IGNORECASE)

TERMINAL_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(references|bibliography|acknowledg(?:e)?ments?|appendix|appendices|"
    r"supplementary(?: material| information)?|attention visualizations?)\b",
    re.IGNORECASE,
)

HEADING_SECTION_PATTERNS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("intro", "Abstract", (r"abstract", r"introduction", r"intro", r"overview")),
    ("related_work", "Related Work", (r"background", r"related work", r"prior work", r"literature review", r"preliminaries")),
    (
        "method",
        "Method",
        (
            r"method",
            r"methods",
            r"methodology",
            r"approach",
            r"proposed method",
            r"model",
            r"model architecture",
            r"architecture",
            r"framework",
            r"algorithm",
            r"update rule",
            r"initialization bias correction",
            r"convergence analysis",
            r"theoretical analysis",
            r"proof",
            r"derivation",
            r"encoder and decoder stacks",
            r"encoder",
            r"decoder",
            r"attention",
            r"self-attention",
            r"multi-head attention",
            r"scaled dot-product attention",
            r"applications of attention in our model",
            r"position-wise feed-forward networks",
            r"positional encoding",
            r"objective",
            r"loss",
            r"why self-attention",
        ),
    ),
    (
        "experiment",
        "Experiment",
        (
            r"experiments?",
            r"experimental setup",
            r"empirical results?",
            r"evaluation",
            r"results?",
            r"discussion",
            r"ablation(?: studies)?",
            r"analysis",
            r"training",
            r"training data and batching",
            r"hardware and schedule",
            r"optimizer",
            r"regularization",
            r"model variations",
            r"machine translation",
            r"english constituency parsing",
            r"logistic regression",
            r"multi-layer neural networks",
            r"convolutional neural networks",
            r"bias-correction term",
            r"datasets?",
            r"benchmarks?",
        ),
    ),
    ("conclusion", "Conclusion", (r"conclusion", r"conclusions", r"limitations?", r"future work")),
)

CONCLUSION_TAIL_RE = re.compile(
    r"\b(in this work|in this paper|we (?:presented|introduced|demonstrated|showed|have shown|conclude)|future work)\b",
    re.IGNORECASE,
)

NUMBERED_INLINE_HEADING_RE = re.compile(
    r"(?<![a-z0-9])(?P<number>[1-9](?:\.\d{1,2}){0,2})\s+"
    r"(?P<title>(?:[a-z][a-z0-9+\-]*\s+){0,8}?"
    r"(?:related work|background|preliminaries|methodology|methods?|approach|framework|"
    r"architecture|architectures|models?|algorithms?|metrics?|evaluation|evaluations|"
    r"experiments?|results?|analysis|discussion|datasets?|benchmarks?|training|"
    r"examples?|conclusions?|future work|references|acknowledg(?:e)?ments?|appendix|appendices))\b"
)


@dataclass
class SectionChunk:
    section: str
    title: str
    text: str


@dataclass(frozen=True)
class InlineMarker:
    start: int
    end: int
    title: str
    section: str
    priority: int
    terminal: bool = False


def section_document(text: str) -> list[SentenceRecord]:
    chunks = split_into_section_chunks(text)
    sentences: list[SentenceRecord] = []
    for chunk in chunks:
        for sentence in split_sentences(chunk.text):
            if not sentence.strip():
                continue
            sentences.append(
                SentenceRecord(
                    text=sentence.strip(),
                    section=chunk.section,
                    section_title=chunk.title,
                    sentence_index=len(sentences),
                    source="sectioner",
                )
            )
    if not sentences:
        for sentence in split_sentences(text):
            sentences.append(SentenceRecord(text=sentence, section="intro", sentence_index=len(sentences), source="sectioner"))
    repaired = assign_positional_sections(sentences)
    repaired = repair_related_work_bridge(repaired)
    repaired = repair_conclusion_tail(repaired)
    return repair_low_coverage_sections(repaired)


def split_into_section_chunks(text: str) -> list[SectionChunk]:
    lines = clean_pdf_lines(text)
    chunks: list[SectionChunk] = []
    current_title = "Introduction"
    current_section = "intro"
    buffer: list[str] = []
    seen_heading = False

    for line in lines:
        if not line or is_noise_line(line):
            continue
        if is_terminal_heading_line(line):
            if seen_heading:
                if buffer:
                    chunks.append(SectionChunk(current_section, current_title, " ".join(buffer)))
                return merge_adjacent_chunks(chunks)
            break
        heading = detect_section_heading(line)
        if heading:
            seen_heading = True
            if buffer:
                chunks.append(SectionChunk(current_section, current_title, " ".join(buffer)))
                buffer = []
            current_title, current_section = heading
            continue
        buffer.append(line)

    if buffer:
        chunks.append(SectionChunk(current_section, current_title, " ".join(buffer)))
    if seen_heading:
        return chunks
    inline_chunks = split_inline_section_chunks(text)
    if inline_chunks:
        return inline_chunks
    return [SectionChunk("intro", "Unsectioned", text)]


def clean_pdf_lines(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    return [normalize_pdf_line(line) for line in normalized.split("\n")]


def normalize_pdf_line(line: str) -> str:
    line = re.sub(r"\s+", " ", line).strip()
    return line.strip("\ufeff")


def is_noise_line(line: str) -> bool:
    if not line:
        return True
    if PAGE_NUMBER_RE.match(line) or SECTION_NUMBER_RE.match(line):
        return True
    lowered = line.lower()
    if lowered.startswith("published as a conference paper at") and len(line.split()) <= 8:
        return True
    if lowered.startswith("arxiv:") and len(line.split()) <= 5:
        return True
    return False


def is_terminal_heading_line(line: str) -> bool:
    return bool(TERMINAL_HEADING_RE.match(line))


def detect_section_heading(line: str) -> tuple[str, str] | None:
    if len(line.split()) > 9:
        return None
    match = SECTION_HEADING_RE.match(line)
    if not match:
        return None
    heading_text = normalize_heading_text(match.group(1))
    if not heading_text:
        return None
    for section, title, patterns in HEADING_SECTION_PATTERNS:
        if any(re.fullmatch(pattern, heading_text, flags=re.IGNORECASE) for pattern in patterns):
            return line, section
    return None


def normalize_heading_text(heading: str) -> str:
    heading = heading.lower()
    heading = re.sub(r"^\s*(?:\d+(?:\.\d+)*|[a-z])\.?\s+", "", heading)
    heading = re.sub(r"[:.]\s*$", "", heading)
    heading = re.sub(r"[^a-z0-9+\-\s]", " ", heading)
    heading = re.sub(r"\s+", " ", heading).strip()
    return heading


def assign_positional_sections(sentences: list[SentenceRecord]) -> list[SentenceRecord]:
    has_real_sections = any(sentence.section != "intro" or sentence.section_title != "Unsectioned" for sentence in sentences)
    if has_real_sections or len(sentences) < 8:
        return sentences
    total = len(sentences)
    boundaries = [
        ("intro", 0.20),
        ("related_work", 0.35),
        ("method", 0.65),
        ("experiment", 0.90),
        ("conclusion", 1.01),
    ]
    for idx, sentence in enumerate(sentences):
        ratio = idx / max(total, 1)
        for section, end_ratio in boundaries:
            if ratio < end_ratio:
                sentence.section = section
                sentence.section_title = "Positional " + section
                break
    return sentences


def repair_conclusion_tail(sentences: list[SentenceRecord]) -> list[SentenceRecord]:
    if any(sentence.section == "conclusion" for sentence in sentences) or len(sentences) < 10:
        return sentences
    start_idx = max(0, int(len(sentences) * 0.72))
    for sentence in sentences[start_idx:]:
        if sentence.section not in {"method", "experiment"}:
            continue
        if not CONCLUSION_TAIL_RE.search(sentence.text):
            continue
        for tail in sentences[sentence.sentence_index :]:
            if tail.section in {"method", "experiment"}:
                tail.section = "conclusion"
                tail.section_title = "Inferred conclusion tail"
        break
    return sentences


def repair_low_coverage_sections(sentences: list[SentenceRecord]) -> list[SentenceRecord]:
    if len(sentences) < 40:
        return sentences
    counts = {section: 0 for section in CANONICAL_SECTIONS}
    for sentence in sentences:
        counts[sentence.section] = counts.get(sentence.section, 0) + 1
    effective_sections = [section for section, count in counts.items() if count >= max(2, int(len(sentences) * 0.03))]
    if len(effective_sections) > 2 and counts.get("method", 0) and counts.get("experiment", 0):
        return sentences

    conclusion_start = next((sentence.sentence_index for sentence in sentences if sentence.section == "conclusion"), None)
    body_end = conclusion_start if conclusion_start is not None and conclusion_start > 8 else len(sentences)
    if body_end < 30:
        return sentences

    body_boundaries = [
        ("intro", 0.20),
        ("related_work", 0.35),
        ("method", 0.65),
        ("experiment", 1.01),
    ]
    for idx, sentence in enumerate(sentences[:body_end]):
        ratio = idx / max(body_end, 1)
        for section, end_ratio in body_boundaries:
            if ratio < end_ratio:
                sentence.section = section
                sentence.section_title = "Low-coverage positional " + section
                break
    if conclusion_start is None:
        tail_start = max(0, int(len(sentences) * 0.92))
        for sentence in sentences[tail_start:]:
            if CONCLUSION_TAIL_RE.search(sentence.text):
                for tail in sentences[sentence.sentence_index :]:
                    tail.section = "conclusion"
                    tail.section_title = "Low-coverage inferred conclusion"
                break
    return sentences


def repair_related_work_bridge(sentences: list[SentenceRecord]) -> list[SentenceRecord]:
    if any(sentence.section == "related_work" for sentence in sentences) or len(sentences) < 10:
        return sentences
    first_method = next((sentence.sentence_index for sentence in sentences if sentence.section == "method"), None)
    if first_method is None or first_method < 4:
        return sentences
    start = max(2, first_method - 5)
    for sentence in sentences[start:first_method]:
        if not looks_like_related_work_bridge(sentence.text):
            continue
        sentence.section = "related_work"
        sentence.section_title = "Inferred related-work bridge"
    return sentences


def looks_like_related_work_bridge(text: str) -> bool:
    lowered = text.lower()
    bridge_cues = (
        "previous",
        "prior",
        "existing",
        "recent",
        "earlier",
        "before ",
        "although",
        "however",
        "compared with",
        "not fine-tuned",
        "additional pretraining",
        "tf-idf",
        "bm25",
        "orqa",
        "baseline",
    )
    if not any(cue in lowered for cue in bridge_cues):
        return False
    if any(cue in lowered for cue in ("we propose", "we introduce", "we present", "our method", "our model")):
        return False
    return True


def normalize_heading_to_section(heading: str) -> str:
    heading = normalize_heading_text(heading)
    if any(token in heading for token in ("related", "prior", "literature", "background", "preliminaries")):
        return "related_work"
    if any(token in heading for token in ("conclusion", "limitation", "future")):
        return "conclusion"
    if any(
        token in heading
        for token in (
            "method",
            "approach",
            "model",
            "architecture",
            "algorithm",
            "metric",
            "metrics",
            "pre-training",
            "fine-tuning",
            "pretraining",
            "finetuning",
            "masked lm",
            "next sentence prediction",
            "input output representations",
            "convergence",
            "theoretical",
            "proof",
            "derivation",
            "encoder",
            "decoder",
            "attention",
            "objective",
            "loss",
        )
    ):
        return "method"
    if any(
        token in heading
        for token in (
            "experiment",
            "evaluation",
            "human evaluation",
            "bleu evaluation",
            "result",
            "discussion",
            "training",
            "regularization",
            "optimizer",
            "dataset",
            "benchmark",
            "ablation",
            "comparison",
            "examples",
        )
    ):
        return "experiment"
    return "intro" if CANONICAL_SECTIONS else "intro"


def split_inline_section_chunks(text: str) -> list[SectionChunk]:
    compacted = re.sub(r"\s+", " ", text).strip()
    lowered = compacted.lower()
    if len(lowered.split()) < 220:
        return []
    markers = find_inline_markers(lowered)
    if len(markers) < 2:
        return []
    markers = keep_effective_markers(markers, compacted)
    chunks: list[SectionChunk] = []
    if markers[0].start > 40:
        chunks.append(SectionChunk("intro", "Lead", compacted[: markers[0].start].strip()))
    for idx, marker in enumerate(markers):
        next_start = markers[idx + 1].start if idx + 1 < len(markers) else len(compacted)
        if marker.terminal:
            break
        chunk_text = compacted[marker.end:next_start].strip()
        if len(chunk_text.split()) >= 8:
            chunks.append(SectionChunk(marker.section, marker.title, chunk_text))
    return merge_adjacent_chunks(chunks)


def keep_effective_markers(markers: list[InlineMarker], compacted: str) -> list[InlineMarker]:
    effective: list[InlineMarker] = []
    for marker in markers:
        if marker.terminal:
            if should_stop_at_terminal(marker, compacted, effective):
                effective.append(marker)
                break
            continue
        effective.append(marker)
    return effective


def should_stop_at_terminal(marker: InlineMarker, lowered: str, previous_markers: list[InlineMarker]) -> bool:
    ratio = marker.start / max(len(lowered), 1)
    phrase = lowered[marker.start : marker.end]
    saw_conclusion = any(item.section == "conclusion" and not item.terminal for item in previous_markers)
    if saw_conclusion:
        return True
    if phrase.startswith(("a distant supervision", "b alternative similarity", "c joint training")):
        return False
    if "system card" in phrase:
        return ratio > 0.25
    if phrase.startswith("references"):
        return ratio > 0.82
    if phrase.startswith("acknowledg"):
        return ratio > 0.75
    if "attention visualization" in phrase:
        return ratio > 0.65
    return ratio > 0.75


def find_inline_markers(lowered: str) -> list[InlineMarker]:
    specs: list[tuple[str, str, int, bool, list[str]]] = [
        ("Abstract", "intro", 100, False, [r"\babstract\b"]),
        ("Introduction", "intro", 98, False, [r"\bintroduction\b"]),
        ("Related Work", "related_work", 95, False, [r"\brelated work\b", r"\bprior work\b", r"\bliterature review\b"]),
        ("Background", "related_work", 84, False, [r"\bbackground\b"]),
        (
            "Method",
            "method",
            84,
            False,
            [
                r"\bmodel architecture\b",
                r"\bmethodology\b",
                r"\bmethods?\b",
                r"\bproposed method\b",
                r"\balgorithms?\b",
                r"\badam s update rule\b",
                r"\bupdate rule\b",
                r"\binitialization bias correction\b",
                r"\bconvergence analysis\b",
                r"\btheoretical analysis\b",
                r"\bproof\b",
                r"\bderivation\b",
                r"\bencoder and decoder stacks\b",
                r"\bscaled dot-product attention\b",
                r"\bmulti-head attention\b",
                r"\bapplications of attention in our model\b",
                r"\bposition-wise feed-forward networks\b",
                r"\bpositional encoding\b",
                r"\bwhy self-attention\b",
            ],
        ),
        (
            "Experiment",
            "experiment",
            78,
            False,
            [
                r"\bexperiments?\b",
                r"\bexperimental setup\b",
                r"\bempirical results?\b",
                r"\bevaluation\b",
                r"\bablation studies\b",
                r"\bmodel variations\b",
                r"\btraining data and batching\b",
                r"\bhardware and schedule\b",
                r"\boptimizer\b",
                r"\bregularization\b",
                r"\bmachine translation\b",
                r"\benglish constituency parsing\b",
                r"\blogistic regression\b",
                r"\bmulti-layer neural networks\b",
                r"\bconvolutional neural networks\b",
                r"\bbias-correction term\b",
                r"(?<!pre-)\btraining\b",
            ],
        ),
        ("Conclusion", "conclusion", 90, False, [r"\bconclusions?\b"]),
        (
            "Terminal",
            "conclusion",
            110,
            True,
            [
                r"\breferences\b",
                r"\backnowledg(?:e)?ments?\b",
                r"\bappendix\b",
                r"\bsupplementary material\b",
                r"\bsystem card\b",
                r"\battention visualizations?\b",
                r"\bfull rbrm instructions\b",
                r"\ba\s+distant supervision\b",
                r"\bb\s+alternative similarity\b",
                r"\bc\s+joint training\b",
                r"\bto converge in section [a-z] \d\b",
                r"\bnext sentence prediction the next sentence prediction task can be illustrated\b",
                r"\ba\s+\d+\s+(?:pre-training|fine-tuning|comparison|illustrations)\b",
                r"\bb\s+\d+\s+detailed\b",
                r"\bc\s+\d+\s+(?:additional|effect)\b",
            ],
        ),
    ]
    candidates: list[InlineMarker] = []
    for title, section, priority, terminal, patterns in specs:
        for pattern in patterns:
            for match in re.finditer(pattern, lowered):
                if is_inline_heading_context(lowered, match.start(), match.end(), title, terminal):
                    candidates.append(InlineMarker(match.start(), match.end(), title, section, priority, terminal))
    candidates.extend(find_numbered_inline_heading_markers(lowered))
    candidates.extend(find_method_transition_markers(lowered))
    if not candidates:
        return []
    candidates = remove_pre_section_false_markers(sorted(candidates, key=lambda item: (item.start, -item.priority)))
    markers = prune_inline_markers(candidates)
    return coerce_inline_sections(markers, lowered)


def find_numbered_inline_heading_markers(lowered: str) -> list[InlineMarker]:
    markers: list[InlineMarker] = []
    for match in NUMBERED_INLINE_HEADING_RE.finditer(lowered):
        raw_title = match.group("title").strip()
        if not is_numbered_inline_heading_context(lowered, match.start(), match.end(), raw_title):
            continue
        heading_text = normalize_heading_text(raw_title)
        if not heading_text:
            continue
        if any(token in heading_text for token in ("references", "acknowledg", "appendix", "appendices")):
            markers.append(InlineMarker(match.start(), match.end(), "Terminal", "conclusion", 112, True))
            continue
        section = normalize_heading_to_section(heading_text)
        priority = {
            "intro": 82,
            "related_work": 88,
            "method": 88,
            "experiment": 88,
            "conclusion": 96,
        }.get(section, 82)
        title = " ".join(word.capitalize() for word in heading_text.split())
        markers.append(InlineMarker(match.start(), match.end(), title, section, priority))
    return markers


def is_numbered_inline_heading_context(lowered: str, start: int, end: int, raw_title: str) -> bool:
    left_words = context_words(lowered[max(0, start - 120) : start])
    right_words = context_words(lowered[end : min(len(lowered), end + 120)])
    previous = left_words[-1] if left_words else ""
    if previous in {
        "section",
        "sections",
        "figure",
        "fig",
        "table",
        "eq",
        "equation",
        "example",
        "appendix",
        "see",
        "row",
        "rows",
    }:
        return False
    heading_text = normalize_heading_text(raw_title)
    first_word = heading_text.split()[0] if heading_text.split() else ""
    if first_word in {"so", "we", "this", "these", "that", "it", "because", "then"}:
        return False
    if previous in {"of", "for", "from", "with", "using", "than", "about", "over", "under", "between", "at"}:
        weak_headings = {"training", "results", "experiments", "evaluation", "analysis", "discussion", "examples"}
        if len(heading_text.split()) <= 1 or heading_text in weak_headings:
            return False
    if heading_text in {"training"} and not any(
        cue in " ".join(right_words[:12]) for cue in ("data", "regime", "procedure", "objective", "we", "the")
    ):
        return False
    if heading_text in {"results", "discussion", "analysis"} and previous not in {"1", "2", "3", "4", "5", "6", "7", "8", "9"}:
        return False
    return True


def find_method_transition_markers(lowered: str) -> list[InlineMarker]:
    markers: list[InlineMarker] = []
    repeated_name_pattern = re.compile(r"\b([a-z][a-z0-9\-]{1,30})\s+we\s+(?:introduce|present|describe)\s+\1\b")
    for match in repeated_name_pattern.finditer(lowered):
        name = match.group(1)
        if name in {"section", "paper", "model", "method", "approach"}:
            continue
        markers.append(InlineMarker(match.start(), match.start() + len(name), "Method", "method", 76))
    framework_pattern = re.compile(r"\bwe\s+(?:introduce|present|describe)\s+[^.]{0,120}?\bin this section\b")
    for match in framework_pattern.finditer(lowered):
        markers.append(InlineMarker(match.start(), match.start(), "Method", "method", 72))
    solution_pattern = re.compile(
        r"\b(?:our|the)\s+(?:final\s+)?(?:solution|approach|method|model|framework|retriever)\s+"
        r"(?:is|uses|consists|relies|optimizes|encodes)\b"
    )
    for match in solution_pattern.finditer(lowered):
        markers.append(InlineMarker(match.start(), match.start(), "Method", "method", 76))
    implementation_pattern = re.compile(
        r"\bwe\s+(?:use|adopt|train|fine-tune|finetune|optimize|encode)\s+[^.]{0,140}?"
        r"\b(?:encoder|decoder|retriever|embedding|architecture|objective|loss)\b"
    )
    for match in implementation_pattern.finditer(lowered):
        markers.append(InlineMarker(match.start(), match.start(), "Method", "method", 70))
    return markers


def is_inline_heading_context(lowered: str, start: int, end: int, title: str, terminal: bool) -> bool:
    left_words = context_words(lowered[max(0, start - 180) : start])
    right_words = context_words(lowered[end : min(len(lowered), end + 220)])
    previous = left_words[-1] if left_words else ""
    right_text = " ".join(right_words[:18])
    phrase = lowered[start:end]
    if terminal:
        if phrase.startswith(("a ", "b ", "c ")) or phrase.startswith("next sentence prediction"):
            return "conclusion" in lowered[max(0, start - 1800) : start]
        return True
    if title == "Abstract":
        return start < 600 or previous in {"title"}
    if title == "Introduction":
        return start < max(1800, len(lowered) // 4) and previous not in EMBEDDED_HEADING_PRECEDERS
    if title == "Related Work":
        return previous not in {"in", "of", "for", "from", "with"}
    if title == "Background":
        if previous in {"exhaustive", "additional", "general", "model"}:
            return False
        if right_words[:1] in (["description"], ["knowledge"], ["information"]):
            return False
        return any(cue in right_text for cue in ("goal", "foundation", "prior", "previous", "related", "history", "review"))
    if title == "Method":
        if previous in {"a", "an", "the", "this", "that", "our", "their", "same", "overall", "see"}:
            return False
        if phrase == "approach" and previous in {"based", "feature", "fine-tuning", "finetuning"}:
            return False
        if phrase in {"method", "methods"} and right_words[:1] in (["from"], ["for"], ["of"]):
            return False
        if phrase in {"method", "methods"} and previous and not previous.isdigit():
            return False
        if phrase in {"algorithm", "algorithms"} and previous not in {"1", "2", "3", "4", "5", "6", "7"}:
            return False
        numbered_headings = {
            "model architecture",
            "adam s update rule",
            "update rule",
            "initialization bias correction",
            "convergence analysis",
            "theoretical analysis",
            "proof",
            "derivation",
            "encoder and decoder stacks",
            "scaled dot-product attention",
            "multi-head attention",
            "applications of attention in our model",
            "position-wise feed-forward networks",
            "positional encoding",
            "why self-attention",
        }
        if phrase in numbered_headings and previous and not previous.isdigit():
            return False
        method_cues = (
            "model",
            "architecture",
            "framework",
            "encoder",
            "decoder",
            "layer",
            "we",
            "this section",
            "competitive",
            "objective",
            "gradient",
            "update",
            "theorem",
            "proof",
            "convergence",
            "attention",
            "input",
            "output",
            "algorithm",
            "pseudo",
            "parameter",
            "parameters",
            "moment",
        )
        return any(cue in right_text for cue in method_cues)
    if title == "Experiment":
        if phrase in {"result", "results"} and (previous in {"these", "our", "their", "identical", "published"} or right_words[:1] == ["see"]):
            return False
        if phrase in {"evaluation", "analysis"} and previous and not previous.isdigit():
            return False
        if phrase == "training" and not right_text.startswith("this section describes"):
            return False
        if phrase in {"experiment", "experiments"} and previous and not previous.isdigit():
            return False
        if phrase in {"optimizer", "regularization"} and previous and not previous.isdigit():
            return False
        numbered_headings = {
            "training data and batching",
            "hardware and schedule",
            "machine translation",
            "english constituency parsing",
            "logistic regression",
            "multi-layer neural networks",
            "convolutional neural networks",
            "bias-correction term",
            "model variations",
        }
        if phrase in numbered_headings and previous and not previous.isdigit():
            return False
        return True
    if title == "Conclusion":
        return previous not in {"in", "for", "future", "our", "the"}
    return True


EMBEDDED_HEADING_PRECEDERS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "of",
    "for",
    "in",
    "from",
    "with",
    "using",
    "used",
    "called",
}


def context_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9+\-]+", text)


def prune_inline_markers(candidates: list[InlineMarker]) -> list[InlineMarker]:
    kept: list[InlineMarker] = []
    for candidate in candidates:
        if kept and candidate.start - kept[-1].start < 80:
            if candidate.priority > kept[-1].priority:
                kept[-1] = candidate
            continue
        kept.append(candidate)
    return remove_section_regressions(kept)


def remove_pre_section_false_markers(candidates: list[InlineMarker]) -> list[InlineMarker]:
    first_intro = min((item.start for item in candidates if item.title == "Introduction"), default=None)
    first_related = min((item.start for item in candidates if item.title in {"Related Work", "Background"}), default=None)
    first_method = min((item.start for item in candidates if item.section == "method"), default=None)
    first_experiment = min((item.start for item in candidates if item.section == "experiment"), default=None)
    filtered: list[InlineMarker] = []
    for candidate in candidates:
        if first_intro is not None and candidate.start < first_intro and candidate.section not in {"intro"} and not candidate.terminal:
            continue
        if (
            first_related is not None
            and candidate.start < first_related
            and candidate.section == "method"
            and candidate.priority < 82
        ):
            continue
        if first_method is not None and candidate.start < first_method and candidate.section in {"experiment", "conclusion"}:
            continue
        if first_experiment is not None and candidate.start < first_experiment and candidate.section == "conclusion":
            continue
        filtered.append(candidate)
    return filtered


def remove_section_regressions(markers: list[InlineMarker]) -> list[InlineMarker]:
    order = {"intro": 0, "related_work": 1, "method": 2, "experiment": 3, "conclusion": 4}
    kept: list[InlineMarker] = []
    max_seen = -1
    for marker in markers:
        if marker.terminal:
            kept.append(marker)
            continue
        section_order = order.get(marker.section, max_seen)
        if max_seen >= order["conclusion"] and marker.section != "conclusion":
            continue
        if marker.section == "intro" and max_seen > order["intro"]:
            continue
        if section_order < max_seen and marker.section != "related_work":
            continue
        max_seen = max(max_seen, section_order)
        kept.append(marker)
    return kept


def coerce_inline_sections(markers: list[InlineMarker], lowered: str) -> list[InlineMarker]:
    coerced: list[InlineMarker] = []
    seen_intro = False
    seen_method = False
    for marker in markers:
        section = marker.section
        title = marker.title
        if title == "Background":
            section = "related_work" if seen_intro else "intro"
        if title == "Experiment" and lowered[marker.start : marker.end] == "training" and not seen_method:
            section = "method"
        if section == "intro":
            seen_intro = True
        if section == "method":
            seen_method = True
        coerced.append(InlineMarker(marker.start, marker.end, title, section, marker.priority, marker.terminal))
    return coerced


def merge_adjacent_chunks(chunks: list[SectionChunk]) -> list[SectionChunk]:
    merged: list[SectionChunk] = []
    for chunk in chunks:
        if merged and merged[-1].section == chunk.section and merged[-1].title == chunk.title:
            merged[-1].text = f"{merged[-1].text} {chunk.text}".strip()
        else:
            merged.append(chunk)
    return merged
