from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_DIR / "src"))

from ske.structure.card import build_paper_card
from ske.structure.evidence_units import build_evidence_units_payload
from ske.structure.schema import CANONICAL_SECTIONS


def parse_args() -> argparse.Namespace:
    lexicon_dir = PROJECT_DIR / "datasets" / "lexicon"
    parser = argparse.ArgumentParser(description="Batch-build Evidence-grounded Concept Units JSON from txt papers.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default=str(PROJECT_DIR / "outputs" / "evidence_units_batch"))
    parser.add_argument("--keyword_checkpoint", default=str(PROJECT_DIR / "models" / "checkpoints" / "keyword_scibert_semeval2010_finetune_nobow"))
    parser.add_argument("--structured_checkpoint", default=str(PROJECT_DIR / "models" / "checkpoints" / "structure_v4_partial_role_balanced_fulldev"))
    parser.add_argument("--bow_csv", default=None)
    parser.add_argument("--section_bow_csv", default=str(lexicon_dir / "section_bow_vocabulary.csv"))
    parser.add_argument("--term_frequency_matrix_csv", default=str(lexicon_dir / "section_document_term_matrix_frequency.csv"))
    parser.add_argument("--term_tfidf_matrix_csv", default=str(lexicon_dir / "section_document_term_matrix_tfidf.csv"))
    parser.add_argument("--evidence_cue_csv", default=str(lexicon_dir / "evidence_cue_lexicon.csv"))
    parser.add_argument("--sentence_evidence_csv", default=str(lexicon_dir / "sentence_evidence_candidates.csv"))
    parser.add_argument("--top_k_keyphrases", type=int, default=28)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--files", nargs="*", default=None)
    parser.add_argument("--include_legacy_card", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = collect_paths(input_dir, args.files, args.limit)

    started = time.time()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for idx, path in enumerate(paths, start=1):
        print(f"[{idx}/{len(paths)}] {path.name}", flush=True)
        try:
            rows.append(process_one(path, output_dir, args))
        except Exception as exc:  # noqa: BLE001 - batch reports should keep going.
            failures.append({"file": path.name, "error": repr(exc)})
            print(f"  failed: {exc!r}", flush=True)

    report = build_report(paths, rows, failures, time.time() - started, output_dir)
    (output_dir / "_batch_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


def collect_paths(input_dir: Path, requested: list[str] | None, limit: int | None) -> list[Path]:
    if requested:
        paths = [(input_dir / name) if not Path(name).is_absolute() else Path(name) for name in requested]
    else:
        paths = sorted(input_dir.glob("*.txt"))
    if limit is not None:
        paths = paths[:limit]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input files: " + "; ".join(missing))
    return paths


def process_one(path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    card = build_paper_card(
        text=text,
        keyword_checkpoint=args.keyword_checkpoint,
        structured_checkpoint=args.structured_checkpoint,
        title=path.stem,
        bow_csv=args.bow_csv,
        top_k_keyphrases=args.top_k_keyphrases,
        device_name=args.device,
    )
    payload = build_evidence_units_payload(
        card=card,
        text=text,
        source_path=path,
        keyword_checkpoint=args.keyword_checkpoint,
        structured_checkpoint=args.structured_checkpoint,
        bow_csv=args.bow_csv,
        section_bow_csv=existing_path_or_none(args.section_bow_csv),
        term_frequency_matrix_csv=existing_path_or_none(args.term_frequency_matrix_csv),
        term_tfidf_matrix_csv=existing_path_or_none(args.term_tfidf_matrix_csv),
        evidence_cue_csv=existing_path_or_none(args.evidence_cue_csv),
        sentence_evidence_csv=existing_path_or_none(args.sentence_evidence_csv),
        include_legacy_card=args.include_legacy_card,
    )
    output_path = output_dir / f"{safe_name(path.stem)}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    section_counts = payload["document"]["section_counts"]
    unit_sections = Counter(unit["location"]["section"] for unit in payload["evidence_units"])
    return {
        "file": path.name,
        "output_json": str(output_path.resolve()),
        "sentence_count": payload["document"]["sentence_count"],
        "section_counts": {section: section_counts.get(section, 0) for section in CANONICAL_SECTIONS},
        "evidence_units": len(payload["evidence_units"]),
        "evidence_unit_sections": {section: unit_sections.get(section, 0) for section in CANONICAL_SECTIONS},
    }


def build_report(paths: list[Path], rows: list[dict[str, Any]], failures: list[dict[str, str]], elapsed: float, output_dir: Path) -> dict[str, Any]:
    present_sections = Counter(sum(1 for value in row["section_counts"].values() if value > 0) for row in rows)
    missing_sections: Counter[str] = Counter()
    for row in rows:
        for section, count in row["section_counts"].items():
            if count == 0:
                missing_sections[section] += 1
    summary = {
        "total": len(paths),
        "ok": len(rows),
        "failed": len(failures),
        "runtime_seconds": round(elapsed, 3),
        "output_dir": str(output_dir.resolve()),
        "present_sections": dict(sorted(present_sections.items())),
        "missing_sections": {section: missing_sections.get(section, 0) for section in CANONICAL_SECTIONS},
        "avg_evidence_units": round(sum(row["evidence_units"] for row in rows) / max(len(rows), 1), 3),
        "min_evidence_units": min((row["evidence_units"] for row in rows), default=0),
        "max_evidence_units": max((row["evidence_units"] for row in rows), default=0),
    }
    return {"summary": summary, "files": rows, "failures": failures}


def existing_path_or_none(value: str | None) -> str | None:
    if not value:
        return None
    return value if Path(value).exists() else None


def safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return safe.strip("_") or "paper"


if __name__ == "__main__":
    main()
