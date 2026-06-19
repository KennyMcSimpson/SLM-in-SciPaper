from __future__ import annotations

from dataclasses import dataclass

from .text_utils import as_keyphrase_list, as_text, split_sentences


@dataclass
class KeyphraseDocument:
    doc_id: str
    title: str
    abstract: str
    full_text: str
    keyphrases: list[str]
    source_tokens: list[str] | None = None
    source_bio_tags: list[str] | None = None
    keyphrase_source: str = "explicit"

    @property
    def text(self) -> str:
        return "\n".join(part for part in (self.title, self.abstract, self.full_text) if part)

    @property
    def sentences(self) -> list[str]:
        return split_sentences(self.text)


def normalize_record(record: dict, fallback_id: str) -> KeyphraseDocument:
    title = as_text(_first_present(record, ("title", "name")))
    abstract = as_text(_first_present(record, ("abstract", "abstract_text", "summary")))
    full_text = as_text(_first_present(record, ("fulltext", "full_text", "document", "text", "article")))
    if not full_text and not abstract:
        source = _first_present(record, ("source", "sentences"))
        full_text = " ".join(as_text(sentence) for sentence in source) if isinstance(source, list) else as_text(source)
    keyphrases = as_keyphrase_list(
        _first_present(
            record,
            (
                "keyphrases",
                "keywords",
                "keyword",
                "target",
                "targets",
                "present_kps",
                "extractive_keyphrases",
                "abstractive_keyphrases",
            ),
        )
    )
    return KeyphraseDocument(
        doc_id=as_text(_first_present(record, ("id", "doc_id", "paper_id"))) or fallback_id,
        title=title,
        abstract=abstract,
        full_text=full_text,
        keyphrases=keyphrases,
    )


def normalize_midas_record(dataset_name: str, record: dict, fallback_id: str) -> KeyphraseDocument:
    """Normalize MIDAS keyphrase datasets without throwing away their real labels."""
    if "document" in record and "doc_bio_tags" in record:
        return normalize_token_bio_record(record, fallback_id)
    if "sec_text" in record and "sections" in record:
        return normalize_ldkp_record(record, fallback_id)
    return normalize_record(record, fallback_id)


def normalize_token_bio_record(record: dict, fallback_id: str) -> KeyphraseDocument:
    tokens = [as_text(token) for token in record.get("document", []) if as_text(token)]
    tags = [as_text(tag) for tag in record.get("doc_bio_tags", [])]
    keyphrases = phrases_from_bio(tokens, tags)
    return KeyphraseDocument(
        doc_id=as_text(_first_present(record, ("id", "doc_id", "paper_id"))) or fallback_id,
        title="",
        abstract="",
        full_text=detokenize(tokens),
        keyphrases=keyphrases,
        source_tokens=tokens,
        source_bio_tags=tags,
        keyphrase_source="doc_bio_tags",
    )


def normalize_ldkp_record(record: dict, fallback_id: str) -> KeyphraseDocument:
    sections = [as_text(section) for section in record.get("sections", [])]
    sec_text = record.get("sec_text", []) or []
    sec_bio_tags = record.get("sec_bio_tags", []) or []

    title_parts: list[str] = []
    abstract_parts: list[str] = []
    full_parts: list[str] = []
    all_tokens: list[str] = []
    all_tags: list[str] = []

    for idx, raw_tokens in enumerate(sec_text):
        tokens = [as_text(token) for token in raw_tokens if as_text(token)]
        text = detokenize(tokens)
        section_name = sections[idx].lower() if idx < len(sections) else ""
        if not text:
            continue
        if section_name == "title":
            title_parts.append(text)
        elif section_name == "abstract":
            abstract_parts.append(text)
        else:
            full_parts.append(text)
        all_tokens.extend(tokens)
        tags = sec_bio_tags[idx] if idx < len(sec_bio_tags) else []
        all_tags.extend(as_text(tag) for tag in tags)

    explicit_phrases = as_keyphrase_list(record.get("extractive_keyphrases")) + as_keyphrase_list(record.get("abstractive_keyphrases"))
    bio_phrases = phrases_from_bio(all_tokens, all_tags)
    keyphrases = unique_phrases(explicit_phrases + bio_phrases)
    return KeyphraseDocument(
        doc_id=as_text(_first_present(record, ("id", "doc_id", "paper_id"))) or fallback_id,
        title=" ".join(title_parts),
        abstract=" ".join(abstract_parts),
        full_text="\n".join(full_parts),
        keyphrases=keyphrases,
        source_tokens=all_tokens,
        source_bio_tags=all_tags,
        keyphrase_source="ldkp_keyphrases+sec_bio_tags",
    )


def phrases_from_bio(tokens: list[str], tags: list[str]) -> list[str]:
    phrases: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            phrase = detokenize(current).strip(" \t\r\n,.;:()[]{}\"'")
            if phrase:
                phrases.append(phrase)
            current.clear()

    for token, tag in zip(tokens, tags):
        label = bio_prefix(tag)
        if label == "B":
            flush()
            current.append(token)
        elif label == "I":
            if not current:
                current.append(token)
            else:
                current.append(token)
        else:
            flush()
    flush()
    return unique_phrases(phrases)


def bio_prefix(tag: object) -> str:
    text = as_text(tag).strip().upper()
    if not text:
        return "O"
    if text.startswith("B"):
        return "B"
    if text.startswith("I"):
        return "I"
    return "O"


def unique_phrases(phrases: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for phrase in phrases:
        cleaned = as_text(phrase).strip()
        key = " ".join(cleaned.lower().split())
        if cleaned and key not in seen:
            seen.add(key)
            output.append(cleaned)
    return output


def detokenize(tokens: list[str]) -> str:
    return " ".join(token for token in tokens if token).strip()


def _first_present(record: dict, keys: tuple[str, ...]) -> object:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None
