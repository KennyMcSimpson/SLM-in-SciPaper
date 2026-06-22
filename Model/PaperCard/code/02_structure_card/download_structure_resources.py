from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_dataset

PROJECT_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR = PROJECT_DIR.parent
sys.path.append(str(PROJECT_DIR / "src"))

from ske.data.text_utils import split_sentences, token_jaccard
from ske.structure.data import write_jsonl
from ske.structure.roles import best_summary_facet_role, contains_evidence, greedy_rouge_oracle_labels, infer_role_from_sentence
from ske.structure.schema import SentenceRecord, StructuredDocument
from ske.structure.sectioning import normalize_heading_to_section, section_document


PUBMED_BASE_URL = "https://raw.githubusercontent.com/Franck-Dernoncourt/pubmed-rct/master"
PUBMED_VARIANTS = {
    "20k": "PubMed_20k_RCT_numbers_replaced_with_at_sign",
    "20k_original": "PubMed_20k_RCT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and normalize structure-aware paper-card resources.")
    parser.add_argument("--core", action="store_true", help="Download PubMed RCT 20k, QASPER, and ACLSum.")
    parser.add_argument("--pubmed_variant", choices=sorted(PUBMED_VARIANTS), default="20k")
    parser.add_argument("--skip_pubmed", action="store_true")
    parser.add_argument("--skip_qasper", action="store_true")
    parser.add_argument("--skip_aclsum", action="store_true")
    parser.add_argument("--limit_per_split", type=int, default=None)
    parser.add_argument("--normalize_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.core and not any([args.skip_pubmed, args.skip_qasper, args.skip_aclsum]):
        args.core = True
    report: dict[str, Any] = {"datasets": {}, "sources": dataset_sources()}
    if args.core and not args.skip_pubmed:
        report["datasets"]["pubmed_rct"] = normalize_pubmed_rct(args.pubmed_variant, args.limit_per_split, args.normalize_only)
    if args.core and not args.skip_qasper:
        report["datasets"]["qasper"] = normalize_qasper(args.limit_per_split)
    if args.core and not args.skip_aclsum:
        report["datasets"]["aclsum"] = normalize_aclsum(args.limit_per_split)
    report_path = PROJECT_DIR / "datasets" / "02_structure_card" / "structure_download_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def dataset_sources() -> dict[str, Any]:
    return {
        "pubmed_rct": {
            "role": "sentence role supervision",
            "source": "https://github.com/Franck-Dernoncourt/pubmed-rct",
            "labels": ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"],
        },
        "qasper": {
            "role": "evidence sentence supervision",
            "source": "https://huggingface.co/datasets/allenai/qasper",
            "fields": ["full_text", "qas.answers.answer.evidence", "qas.answers.answer.highlighted_evidence"],
        },
        "aclsum": {
            "role": "facet-to-card weak role and importance supervision",
            "source": "https://huggingface.co/datasets/sobamchan/aclsum",
            "facets": ["challenge", "approach", "outcome"],
        },
        "facetsum": {
            "role": "future optional facet summary supervision",
            "source": "https://huggingface.co/datasets/allenai/facetsum",
            "status": "manual approval / large download; not default core resource",
        },
    }


def normalize_pubmed_rct(variant: str, limit_per_split: int | None, normalize_only: bool) -> dict[str, Any]:
    raw_dir = PROJECT_DIR / "datasets" / "02_structure_card" / "raw" / "pubmed_rct" / variant
    processed_dir = PROJECT_DIR / "datasets" / "02_structure_card" / "processed" / "pubmed_rct"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    folder = PUBMED_VARIANTS[variant]
    split_files = {"train": "train.txt", "validation": "dev.txt", "test": "test.txt"}
    if not normalize_only:
        for filename in split_files.values():
            target = raw_dir / filename
            if not target.exists():
                urllib.request.urlretrieve(f"{PUBMED_BASE_URL}/{folder}/{filename}", target)
    summary: dict[str, Any] = {}
    for split_name, filename in split_files.items():
        docs = parse_pubmed_file(raw_dir / filename, split_name, limit_per_split)
        write_jsonl(processed_dir / f"{split_name}.jsonl", [doc.to_dict() for doc in docs])
        summary[split_name] = split_summary(docs)
    (processed_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "ok", "variant": variant, "raw_path": str(raw_dir), "processed_path": str(processed_dir), "summary": summary}


def parse_pubmed_file(path: Path, split_name: str, limit_docs: int | None) -> list[StructuredDocument]:
    docs: list[StructuredDocument] = []
    current: list[SentenceRecord] = []
    current_id = ""
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("###"):
                if current:
                    docs.append(
                        StructuredDocument(
                            doc_id=current_id or f"pubmed-rct-{split_name}-{len(docs)}",
                            sentences=current,
                            source="pubmed_rct",
                        )
                    )
                    if limit_docs is not None and len(docs) >= limit_docs:
                        return docs
                current = []
                current_id = line.lstrip("#")
                continue
            if "\t" not in line:
                continue
            raw_label, sentence = line.split("\t", 1)
            section, role = map_pubmed_label(raw_label)
            current.append(
                SentenceRecord(
                    text=sentence,
                    section=section,
                    role=role,
                    evidence_label=None,
                    importance_label=None,
                    sentence_index=len(current),
                    source="pubmed_rct",
                )
            )
    if current and (limit_docs is None or len(docs) < limit_docs):
        docs.append(StructuredDocument(doc_id=current_id or f"pubmed-rct-{split_name}-{len(docs)}", sentences=current, source="pubmed_rct"))
    return docs


def map_pubmed_label(label: str) -> tuple[str, str]:
    mapping = {
        "BACKGROUND": ("intro", "background"),
        "OBJECTIVE": ("intro", "objective"),
        "METHODS": ("method", "process"),
        "RESULTS": ("experiment", "result"),
        "CONCLUSIONS": ("conclusion", "finding"),
    }
    return mapping.get(label.upper(), ("intro", "background"))


def normalize_qasper(limit_per_split: int | None) -> dict[str, Any]:
    processed_dir = PROJECT_DIR / "datasets" / "02_structure_card" / "processed" / "qasper"
    processed_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("allenai/qasper")
    assert isinstance(dataset, DatasetDict)
    summary: dict[str, Any] = {}
    for split_name, split in dataset.items():
        max_rows = len(split) if limit_per_split is None else min(len(split), limit_per_split)
        docs = [
            doc
            for idx in range(max_rows)
            if (doc := qasper_record_to_document(dict(split[int(idx)]), split_name, idx)).sentences
        ]
        write_jsonl(processed_dir / f"{split_name}.jsonl", [doc.to_dict() for doc in docs])
        summary[split_name] = split_summary(docs)
    (processed_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "ok", "dataset_id": "allenai/qasper", "processed_path": str(processed_dir), "summary": summary}


def qasper_record_to_document(record: dict[str, Any], split_name: str, idx: int) -> StructuredDocument:
    evidence_strings = collect_qasper_evidence(record.get("qas") or {})
    sentences: list[SentenceRecord] = []
    full_text = record.get("full_text") or {}
    section_names = full_text.get("section_name") or []
    paragraphs_by_section = full_text.get("paragraphs") or []
    for section_name, paragraphs in zip(section_names, paragraphs_by_section):
        section = normalize_heading_to_section(str(section_name))
        if isinstance(paragraphs, str):
            paragraphs = [paragraphs]
        for paragraph in paragraphs or []:
            for sentence in split_sentences(str(paragraph)):
                evidence = 1.0 if contains_evidence(sentence, evidence_strings) else 0.0
                role, _ = infer_role_from_sentence(sentence, section)
                sentences.append(
                    SentenceRecord(
                        text=sentence,
                        section=section,
                        role=role,
                        evidence_label=evidence,
                        importance_label=None,
                        section_title=str(section_name),
                        sentence_index=len(sentences),
                        source="qasper",
                    )
                )
    return StructuredDocument(
        doc_id=str(record.get("id") or f"qasper-{split_name}-{idx}"),
        title=str(record.get("title") or ""),
        abstract=str(record.get("abstract") or ""),
        sentences=sentences,
        source="qasper",
    )


def collect_qasper_evidence(qas: dict[str, Any]) -> set[str]:
    evidence: set[str] = set()
    for answer_group in qas.get("answers") or []:
        if not isinstance(answer_group, dict):
            continue
        answer_payloads = answer_group.get("answer") or []
        if isinstance(answer_payloads, dict):
            answer_payloads = [answer_payloads]
        for answer in answer_payloads:
            if not isinstance(answer, dict):
                continue
            for key in ("evidence", "highlighted_evidence", "extractive_spans"):
                for value in answer.get(key) or []:
                    if str(value).strip():
                        evidence.add(str(value).strip())
    return evidence


def normalize_aclsum(limit_per_split: int | None) -> dict[str, Any]:
    processed_dir = PROJECT_DIR / "datasets" / "02_structure_card" / "processed" / "aclsum"
    processed_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("sobamchan/aclsum", trust_remote_code=True)
    assert isinstance(dataset, DatasetDict)
    summary: dict[str, Any] = {}
    for split_name, split in dataset.items():
        max_rows = len(split) if limit_per_split is None else min(len(split), limit_per_split)
        docs = [
            doc
            for idx in range(max_rows)
            if (doc := aclsum_record_to_document(dict(split[int(idx)]), split_name, idx)).sentences
        ]
        write_jsonl(processed_dir / f"{split_name}.jsonl", [doc.to_dict() for doc in docs])
        summary[split_name] = split_summary(docs)
    (processed_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "ok", "dataset_id": "sobamchan/aclsum", "processed_path": str(processed_dir), "summary": summary}


def aclsum_record_to_document(record: dict[str, Any], split_name: str, idx: int) -> StructuredDocument:
    summaries = {
        "challenge": str(record.get("challenge") or ""),
        "approach": str(record.get("approach") or ""),
        "outcome": str(record.get("outcome") or ""),
    }
    sentences: list[SentenceRecord] = []
    sectioned_sentences = section_document(str(record.get("document") or ""))
    oracle_importance = greedy_rouge_oracle_labels([sentence.text for sentence in sectioned_sentences], summaries)
    for sentence, importance in zip(sectioned_sentences, oracle_importance):
        role, confidence = best_summary_facet_role(sentence.text, summaries)
        if role is None or confidence < 0.08:
            role, confidence = infer_role_from_sentence(sentence.text, sentence.section)
        else:
            confidence = min(1.0, 0.55 + confidence)
        sentences.append(
            SentenceRecord(
                text=sentence.text,
                section=sentence.section,
                role=role,
                evidence_label=None,
                importance_label=importance,
                section_title=sentence.section_title,
                sentence_index=len(sentences),
                source="aclsum",
            )
        )
    return StructuredDocument(
        doc_id=str(record.get("id") or f"aclsum-{split_name}-{idx}"),
        title="",
        sentences=sentences,
        summaries=summaries,
        source="aclsum",
    )


def split_summary(docs: list[StructuredDocument]) -> dict[str, Any]:
    sentence_count = sum(len(doc.sentences) for doc in docs)
    evidence_count = sum(1 for doc in docs for sentence in doc.sentences if sentence.evidence_label and sentence.evidence_label >= 0.5)
    importance_labeled = sum(1 for doc in docs for sentence in doc.sentences if sentence.importance_label is not None)
    importance_positive = sum(1 for doc in docs for sentence in doc.sentences if sentence.importance_label and sentence.importance_label >= 0.5)
    role_counts: dict[str, int] = {}
    section_counts: dict[str, int] = {}
    for doc in docs:
        for sentence in doc.sentences:
            if sentence.role:
                role_counts[sentence.role] = role_counts.get(sentence.role, 0) + 1
            section_counts[sentence.section] = section_counts.get(sentence.section, 0) + 1
    return {
        "documents": len(docs),
        "sentences": sentence_count,
        "evidence_positive_sentences": evidence_count,
        "importance_labeled_sentences": importance_labeled,
        "importance_positive_sentences": importance_positive,
        "role_counts": dict(sorted(role_counts.items())),
        "section_counts": dict(sorted(section_counts.items())),
    }


if __name__ == "__main__":
    main()

