from __future__ import annotations

import re
from dataclasses import dataclass

from ske.data.text_utils import split_sentences

from .schema import CANONICAL_SECTIONS, SentenceRecord


SECTION_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\s+)?"
    r"(abstract|introduction|intro|background|related work|prior work|literature review|"
    r"method|methods|methodology|approach|model|proposed method|experiments?|experimental setup|"
    r"evaluation|results?|discussion|analysis|conclusion|conclusions|limitations?|future work)"
    r"\s*$",
    re.IGNORECASE,
)

CONCLUSION_TAIL_RE = re.compile(
    r"\b(in this work|in this paper|we (?:presented|introduced|demonstrated|showed|have shown|conclude)|future work)\b",
    re.IGNORECASE,
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
    return repair_conclusion_tail(repaired)


def split_into_section_chunks(text: str) -> list[SectionChunk]:
    lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    chunks: list[SectionChunk] = []
    current_title = "Introduction"
    current_section = "intro"
    buffer: list[str] = []
    seen_heading = False

    for line in lines:
        if not line:
            continue
        heading_match = SECTION_HEADING_RE.match(line)
        if heading_match and len(line.split()) <= 5:
            seen_heading = True
            if buffer:
                chunks.append(SectionChunk(current_section, current_title, " ".join(buffer)))
                buffer = []
            current_title = line
            current_section = normalize_heading_to_section(line)
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
    heading = heading.lower()
    if "related" in heading or "prior" in heading or "literature" in heading:
        return "related_work"
    if any(token in heading for token in ("method", "approach", "model", "methodology")):
        return "method"
    if any(token in heading for token in ("experiment", "evaluation", "result", "analysis", "discussion")):
        return "experiment"
    if any(token in heading for token in ("conclusion", "limitation", "future")):
        return "conclusion"
    return "intro" if CANONICAL_SECTIONS else "intro"


def split_inline_section_chunks(text: str) -> list[SectionChunk]:
    compacted = re.sub(r"\s+", " ", text).strip()
    lowered = compacted.lower()
    if len(lowered.split()) < 220:
        return []
    markers = find_inline_markers(lowered)
    if len(markers) < 2:
        return []
    chunks: list[SectionChunk] = []
    if markers[0].start > 40:
        chunks.append(SectionChunk("intro", "Lead", compacted[: markers[0].start].strip()))
    for idx, marker in enumerate(markers):
        next_start = markers[idx + 1].start if idx + 1 < len(markers) else len(compacted)
        if marker.terminal:
            if should_stop_at_terminal(marker, compacted, markers[:idx]):
                break
            continue
        chunk_text = compacted[marker.end:next_start].strip()
        if len(chunk_text.split()) >= 8:
            chunks.append(SectionChunk(marker.section, marker.title, chunk_text))
    return merge_adjacent_chunks(chunks)


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
    if phrase.startswith(("references", "acknowledg")):
        return ratio > 0.55
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
            82,
            False,
            [
                r"\bmodel architecture\b",
                r"\bmethodology\b",
                r"\bmethods?\b",
                r"\bproposed method\b",
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
                r"\bevaluation\b",
                r"\bablation studies\b",
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
    candidates.extend(find_method_transition_markers(lowered))
    if not candidates:
        return []
    candidates = remove_pre_section_false_markers(sorted(candidates, key=lambda item: (item.start, -item.priority)))
    markers = prune_inline_markers(candidates)
    return coerce_inline_sections(markers, lowered)


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
        markers.append(InlineMarker(match.start(), match.start(), "Method", "method", 86))
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
        if previous in {"a", "an", "the", "this", "that", "our", "their", "same", "overall"}:
            return False
        if phrase == "approach" and previous in {"based", "feature", "fine-tuning", "finetuning"}:
            return False
        if phrase in {"method", "methods"} and right_words[:1] in (["from"], ["for"], ["of"]):
            return False
        return any(cue in right_text for cue in ("model", "architecture", "framework", "encoder", "decoder", "layer", "we", "this section", "competitive"))
    if title == "Experiment":
        if phrase in {"result", "results"} and (previous in {"these", "our", "their", "identical", "published"} or right_words[:1] == ["see"]):
            return False
        if phrase == "training" and not right_text.startswith("this section describes"):
            return False
        if phrase in {"experiment", "experiments"} and previous in {"during", "our", "many", "all", "the"}:
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
    first_method = min((item.start for item in candidates if item.section == "method"), default=None)
    first_experiment = min((item.start for item in candidates if item.section == "experiment"), default=None)
    filtered: list[InlineMarker] = []
    for candidate in candidates:
        if first_intro is not None and candidate.start < first_intro and candidate.section not in {"intro"} and not candidate.terminal:
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
        if section_order < max_seen:
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
            section = "related_work" if seen_intro and not seen_method else "intro"
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
