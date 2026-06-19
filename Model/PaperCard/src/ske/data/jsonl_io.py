from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import KeyphraseDocument


def document_to_payload(doc: KeyphraseDocument, include_source: bool = False) -> dict:
    payload = {
        "doc_id": doc.doc_id,
        "title": doc.title,
        "abstract": doc.abstract,
        "full_text": doc.full_text,
        "keyphrases": doc.keyphrases,
        "keyphrase_source": doc.keyphrase_source,
    }
    if include_source and doc.source_tokens is not None:
        payload["source_tokens"] = doc.source_tokens
    if include_source and doc.source_bio_tags is not None:
        payload["source_bio_tags"] = doc.source_bio_tags
    return payload


def write_documents(path: str | Path, docs: Iterable[KeyphraseDocument], include_source: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for doc in docs:
            payload = document_to_payload(doc, include_source=include_source)
            handle.write(
                json.dumps(payload, ensure_ascii=False) + "\n"
            )
