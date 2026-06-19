from __future__ import annotations

import re
from collections.abc import Iterable

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
NORMALIZED_TOKEN_CHAR_RE = re.compile(r"[a-z0-9_+\-]")


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9_+\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text: str, fallback_words: int = 80) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    pieces = [piece.strip() for piece in SENTENCE_SPLIT_RE.split(text) if piece.strip()]
    if len(pieces) <= 1 and len(text.split()) > fallback_words * 2:
        return chunk_by_words(text, fallback_words)
    return pieces


def chunk_by_words(text: str, chunk_words: int) -> list[str]:
    words = text.split()
    return [" ".join(words[idx : idx + chunk_words]) for idx in range(0, len(words), chunk_words)]


def as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(as_text(item) for item in value if as_text(item))
    return str(value)


def as_keyphrase_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = value.split(";") if ";" in value else value.split("|") if "|" in value else [value]
        return [part.strip() for part in parts if part.strip()]
    if isinstance(value, Iterable):
        phrases: list[str] = []
        for item in value:
            if isinstance(item, str):
                if item.strip():
                    phrases.append(item.strip())
            elif isinstance(item, Iterable):
                joined = " ".join(str(piece) for piece in item).strip()
                if joined:
                    phrases.append(joined)
            else:
                phrases.append(str(item).strip())
        return [phrase for phrase in phrases if phrase]
    return [str(value).strip()]


def find_normalized_phrase_spans(text: str, phrase: str) -> list[tuple[int, int]]:
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return []
    normalized_text, mapping = normalize_with_char_mapping(text)
    if not normalized_text:
        return []
    pattern = re.compile(r"(?<![a-z0-9_+\-])" + re.escape(normalized_phrase) + r"(?![a-z0-9_+\-])")
    spans: list[tuple[int, int]] = []
    for match in pattern.finditer(normalized_text):
        start_idx = match.start()
        end_idx = match.end() - 1
        if start_idx < len(mapping) and end_idx < len(mapping):
            spans.append((mapping[start_idx][0], mapping[end_idx][1]))
    return spans


def normalize_with_char_mapping(text: str) -> tuple[str, list[tuple[int, int]]]:
    chars: list[str] = []
    mapping: list[tuple[int, int]] = []
    previous_was_space = True
    for idx, raw_char in enumerate(text.lower()):
        char = raw_char if NORMALIZED_TOKEN_CHAR_RE.fullmatch(raw_char) else " "
        if char.isspace():
            if chars and not previous_was_space:
                chars.append(" ")
                mapping.append((idx, idx + 1))
            previous_was_space = True
            continue
        chars.append(char)
        mapping.append((idx, idx + 1))
        previous_was_space = False
    while chars and chars[-1] == " ":
        chars.pop()
        mapping.pop()
    return "".join(chars), mapping


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(normalize_text(left).split())
    right_tokens = set(normalize_text(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
