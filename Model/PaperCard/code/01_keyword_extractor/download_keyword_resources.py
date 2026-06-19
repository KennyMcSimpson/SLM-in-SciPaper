from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoModel, AutoTokenizer

PROJECT_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR = PROJECT_DIR.parent
sys.path.append(str(PROJECT_DIR / "src"))

from ske.data.jsonl_io import document_to_payload, write_documents
from ske.data.schema import KeyphraseDocument, normalize_midas_record, normalize_record
from ske.data.text_utils import normalize_text, split_sentences

CORE_DATASETS = {
    "kp20k": ["midas/kp20k"],
    "ldkp10k": ["midas/ldkp10k", "midas/ldkp3k"],
    "semeval2010": ["midas/semeval2010"],
}
OPTIONAL_DATASETS = {
    "inspec": ["midas/inspec"],
    "krapivin": ["midas/krapivin"],
    "nus": ["midas/nus"],
    "openkp": ["midas/openkp"],
}
DEFAULT_BOW = None
DEFAULT_SCITLDR = PROJECT_DIR / "datasets" / "01_keyword_keyphrase" / "raw" / "scitldr" / "SciTLDR-Data"
DEFAULT_MODEL = "allenai/scibert_scivocab_uncased"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and normalize resources for the keyphrase extractor.")
    parser.add_argument("--core", action="store_true")
    parser.add_argument("--optional", action="store_true")
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--bow_csv", default=None)
    parser.add_argument("--scitldr_dir", default=str(DEFAULT_SCITLDR))
    parser.add_argument("--skip_model", action="store_true")
    parser.add_argument("--normalize_only", action="store_true", help="Rebuild processed JSONL from existing raw datasets.")
    parser.add_argument("--limit_per_split", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.core and not args.optional:
        args.core = True
    report: dict[str, Any] = {"model": None, "datasets": {}, "bow_vocab": None, "scitldr": None}
    if args.normalize_only:
        report["model"] = {"status": "skipped", "reason": "normalize_only"}
    elif not args.skip_model:
        report["model"] = download_model(args.model_name)
    if args.core:
        for name, candidates in CORE_DATASETS.items():
            if args.normalize_only:
                report["datasets"][name] = normalize_cached_dataset(name, args.limit_per_split)
            else:
                report["datasets"][name] = download_dataset_with_fallback(name, candidates, args.limit_per_split)
        report["bow_vocab"] = copy_bow_vocab(Path(args.bow_csv)) if args.bow_csv else {"status": "skipped", "reason": "no bow_csv"}
        report["scitldr"] = normalize_local_scitldr(Path(args.scitldr_dir), args.limit_per_split)
    if args.optional:
        for name, candidates in OPTIONAL_DATASETS.items():
            if args.normalize_only:
                report["datasets"][name] = normalize_cached_dataset(name, args.limit_per_split)
            else:
                report["datasets"][name] = download_dataset_with_fallback(name, candidates, args.limit_per_split)
    report_path = PROJECT_DIR / "datasets" / "01_keyword_keyphrase" / "download_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def download_model(model_name: str) -> dict[str, Any]:
    output_dir = PROJECT_DIR / "models" / "base_scibert" / model_name.replace("/", "_")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModel.from_pretrained(model_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(output_dir)
        model.save_pretrained(output_dir)
        return {"status": "ok", "model_name": model_name, "path": str(output_dir)}
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "model_name": model_name, "error": repr(exc)}


def download_dataset_with_fallback(name: str, candidates: list[str], limit_per_split: int | None) -> dict[str, Any]:
    errors: list[str] = []
    for dataset_id in candidates:
        try:
            raw_dir = PROJECT_DIR / "datasets" / "01_keyword_keyphrase" / "raw" / name
            processed_dir = PROJECT_DIR / "datasets" / "01_keyword_keyphrase" / "processed" / name
            dataset = load_dataset(dataset_id, trust_remote_code=True)
            if not isinstance(dataset, DatasetDict):
                dataset = DatasetDict({"train": dataset})
            raw_dir.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(raw_dir)
            summary = normalize_hf_dataset(name, raw_dir, processed_dir, limit_per_split)
            return {
                "status": "ok",
                "dataset_id": dataset_id,
                "raw_path": str(raw_dir),
                "processed_path": str(processed_dir),
                "splits": list(dataset.keys()),
                "summary": summary,
            }
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{dataset_id}: {exc!r}")
    return {"status": "failed", "candidates": candidates, "errors": errors}


def normalize_cached_dataset(name: str, limit_per_split: int | None) -> dict[str, Any]:
    raw_dir = PROJECT_DIR / "datasets" / "01_keyword_keyphrase" / "raw" / name
    processed_dir = PROJECT_DIR / "datasets" / "01_keyword_keyphrase" / "processed" / name
    if not raw_dir.exists():
        return {"status": "failed", "raw_path": str(raw_dir), "error": "raw dataset does not exist"}
    try:
        summary = normalize_hf_dataset(name, raw_dir, processed_dir, limit_per_split)
        return {
            "status": "ok",
            "raw_path": str(raw_dir),
            "processed_path": str(processed_dir),
            "splits": list(summary.keys()),
            "summary": summary,
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "raw_path": str(raw_dir), "error": repr(exc)}


def normalize_hf_dataset(name: str, raw_dir: Path, processed_dir: Path, limit_per_split: int | None) -> dict[str, Any]:
    dataset = load_from_disk(raw_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    split_map = {"valid": "validation", "val": "validation", "dev": "validation"}
    summary: dict[str, Any] = {}
    for split_name, split in dataset.items():
        output_split = split_map.get(split_name, split_name)
        output_path = processed_dir / f"{output_split}.jsonl"
        preview_path = processed_dir / f"{output_split}_source_preview.json"
        max_rows = len(split) if limit_per_split is None else min(len(split), limit_per_split)
        doc_count = 0
        with_keyphrases = 0
        total_keyphrases = 0
        with_source_bio = 0
        keyphrase_sources: set[str] = set()
        source_previews: list[dict[str, Any]] = []
        with output_path.open("w", encoding="utf-8") as handle:
            for idx in range(max_rows):
                doc = normalize_midas_record(name, dict(split[int(idx)]), fallback_id=f"{name}-{output_split}-{idx}")
                handle.write(json.dumps(document_to_payload(doc), ensure_ascii=False) + "\n")
                doc_count += 1
                with_keyphrases += int(bool(doc.keyphrases))
                total_keyphrases += len(doc.keyphrases)
                with_source_bio += int(bool(doc.source_bio_tags))
                keyphrase_sources.add(doc.keyphrase_source)
                if len(source_previews) < 5 and doc.source_bio_tags:
                    source_previews.append(document_to_payload(doc, include_source=True))
        preview_path.write_text(json.dumps(source_previews, ensure_ascii=False, indent=2), encoding="utf-8")
        summary[output_split] = {
            "documents": doc_count,
            "with_keyphrases": with_keyphrases,
            "avg_keyphrases": total_keyphrases / max(doc_count, 1),
            "with_source_bio": with_source_bio,
            "keyphrase_sources": sorted(keyphrase_sources),
            "source_preview": str(preview_path),
        }
    (processed_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def copy_bow_vocab(source: Path) -> dict[str, Any]:
    target = PROJECT_DIR / "datasets" / "01_keyword_keyphrase" / "vocab" / "final_bow_vocabulary.csv"
    if not source.exists():
        return {"status": "failed", "source": str(source), "error": "source file does not exist"}
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {"status": "ok", "source": str(source), "path": str(target)}


def normalize_local_scitldr(scitldr_dir: Path, limit_per_split: int | None) -> dict[str, Any]:
    if not scitldr_dir.exists():
        return {"status": "failed", "path": str(scitldr_dir), "error": "SciTLDR directory does not exist"}
    processed_dir = PROJECT_DIR / "datasets" / "01_keyword_keyphrase" / "processed" / "scitldr_weak"
    processed_dir.mkdir(parents=True, exist_ok=True)
    source_dir = scitldr_dir / "SciTLDR-FullText"
    if not source_dir.exists():
        source_dir = scitldr_dir / "SciTLDR-AIC"
    summary: dict[str, Any] = {}
    for split_name in ("train", "dev", "validation", "test"):
        path = source_dir / f"{split_name}.jsonl"
        if not path.exists():
            continue
        output_split = "validation" if split_name in {"dev", "validation"} else split_name
        docs = []
        with path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if limit_per_split is not None and idx >= limit_per_split:
                    break
                if not line.strip():
                    continue
                record = json.loads(line)
                source = record.get("source") or record.get("article") or record.get("sentences") or []
                full_text = " ".join(str(item) for item in source) if isinstance(source, list) else str(source)
                title = str(record.get("title") or "")
                target = record.get("target") or record.get("tldr") or record.get("summary") or ""
                target_text = " ".join(target) if isinstance(target, list) else str(target)
                docs.append(
                    KeyphraseDocument(
                        doc_id=str(record.get("paper_id") or record.get("id") or f"scitldr-{output_split}-{idx}"),
                        title=title,
                        abstract="",
                        full_text=full_text,
                        keyphrases=weak_phrases_from_text(title + " " + target_text),
                    )
                )
        write_documents(processed_dir / f"{output_split}.jsonl", docs)
        summary[output_split] = {"documents": len(docs), "weak_keyphrase_source": "title+tldr"}
    (processed_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "ok", "source_path": str(source_dir), "processed_path": str(processed_dir)}


STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "we", "with"}


def weak_phrases_from_text(text: str, max_ngram: int = 4) -> list[str]:
    phrases: set[str] = set()
    for sentence in split_sentences(text) or [text]:
        tokens = [token for token in normalize_text(sentence).split() if token and token not in STOPWORDS]
        for start in range(len(tokens)):
            for end in range(start + 1, min(len(tokens), start + max_ngram) + 1):
                phrase = " ".join(tokens[start:end])
                if len(phrase) >= 4:
                    phrases.add(phrase)
    return sorted(phrases, key=lambda item: (-len(item.split()), item))[:24]


if __name__ == "__main__":
    main()

